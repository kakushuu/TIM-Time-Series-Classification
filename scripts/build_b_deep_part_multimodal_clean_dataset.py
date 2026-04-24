#!/usr/bin/env python3
"""Build a cleaned, self-contained B_deep_part multimodal dataset.

The source metadata comes from the existing aligned full-audio dataset. This
script keeps the train/val/test split day-wise to avoid leakage, removes a very
small set of clearly bad rows, normalizes minor format issues, and materializes
one unified dataset directory that contains:

- train.csv / val.csv / test.csv / all.csv
- per-date CSV exports
- frame JPEGs
- 1-second WAV clips
- symlinks to source MP4 videos
- copied trajectory XLSX parts for provenance
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_CSV = PROJECT_ROOT / "data" / "b_deep_part_audio_1s_20241018_29" / "all.csv"
DEFAULT_PART_DIR = PROJECT_ROOT / "data" / "B_deep_part"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "b_deep_part_multimodal_full_clean_20260417"

TRAIN_DATES = ["2024-10-18", "2024-10-19", "2024-10-20", "2024-10-22", "2024-10-23", "2024-10-24", "2024-10-25"]
VAL_DATES = ["2024-10-26", "2024-10-27"]
TEST_DATES = ["2024-10-28", "2024-10-29"]

REQUIRED_COLUMNS = [
    "frame_path",
    "frame_time",
    "timestamp_inferred",
    "video_file",
    "frame_number",
    "second_in_video",
    "经度",
    "纬度",
    "定位时间",
    "速度",
    "方向角",
    "深度",
    "did",
    "分类",
    "时间戳",
    "source_video_path",
    "audio_start_second",
    "audio_duration_seconds",
    "audio_path",
    "audio_exists",
    "audio_is_silence_fill",
]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def project_relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def parse_dates(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def haversine_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    radius_m = 6371000.0
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius_m * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def hardlink_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "existing"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def symlink_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "existing"
    try:
        os.symlink(src, dst)
        return "symlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def load_source(source_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(source_csv, encoding="utf-8-sig")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"missing required columns in {source_csv}: {missing}")
    return df


def clean_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cleaned = df.copy()
    summary: dict[str, int | float] = {
        "source_rows": int(len(cleaned)),
    }

    cleaned["frame_time"] = pd.to_datetime(cleaned["frame_time"], errors="coerce")
    cleaned["定位时间"] = pd.to_datetime(cleaned["定位时间"], errors="coerce")
    numeric_cols = ["frame_number", "second_in_video", "经度", "纬度", "速度", "方向角", "深度", "did", "分类", "时间戳", "audio_start_second", "audio_duration_seconds"]
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    before = len(cleaned)
    cleaned = cleaned.dropna(subset=["frame_time", "定位时间", "frame_path", "audio_path", "video_file", "source_video_path"])
    summary["dropped_missing_required"] = int(before - len(cleaned))

    before = len(cleaned)
    cleaned = cleaned[cleaned["分类"].between(0, 10)]
    summary["dropped_invalid_label"] = int(before - len(cleaned))

    before = len(cleaned)
    cleaned = cleaned[cleaned["audio_exists"].fillna(False)]
    summary["dropped_missing_audio"] = int(before - len(cleaned))

    direction_eq_360 = cleaned["方向角"].eq(360).sum()
    cleaned.loc[cleaned["方向角"].eq(360), "方向角"] = 0.0
    summary["normalized_direction_360_to_0"] = int(direction_eq_360)

    before = len(cleaned)
    cleaned = cleaned[cleaned["方向角"].between(0, 359.999999, inclusive="both")]
    summary["dropped_invalid_direction"] = int(before - len(cleaned))

    before = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["video_file", "second_in_video"], keep="first")
    summary["dropped_duplicate_video_second"] = int(before - len(cleaned))

    before = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["frame_path"], keep="first")
    summary["dropped_duplicate_frame_path"] = int(before - len(cleaned))

    cleaned = cleaned.sort_values(["video_file", "second_in_video", "frame_time", "frame_path"]).reset_index(drop=True)

    same_prev = cleaned["video_file"].eq(cleaned["video_file"].shift())
    sec_gap = cleaned["second_in_video"] - cleaned["second_in_video"].shift()
    lat_prev = np.radians(cleaned["纬度"].shift().fillna(cleaned["纬度"]).to_numpy())
    lon_prev = np.radians(cleaned["经度"].shift().fillna(cleaned["经度"]).to_numpy())
    lat_cur = np.radians(cleaned["纬度"].to_numpy())
    lon_cur = np.radians(cleaned["经度"].to_numpy())
    dist_prev = haversine_m(lat_prev, lon_prev, lat_cur, lon_cur)
    gps_speed = dist_prev / sec_gap.replace(0, np.nan).to_numpy()
    gps_speed = pd.Series(gps_speed).where(same_prev & sec_gap.eq(1))

    extreme_motion = gps_speed.gt(np.maximum(12.0, cleaned["速度"] * 1.5 + 2.0))
    summary["dropped_extreme_motion_rows"] = int(extreme_motion.fillna(False).sum())
    cleaned = cleaned.loc[~extreme_motion.fillna(False)].copy()

    cleaned["frame_time"] = cleaned["frame_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    cleaned["定位时间"] = cleaned["定位时间"].dt.strftime("%Y-%m-%d %H:%M:%S")
    cleaned["frame_number"] = cleaned["frame_number"].astype(int)
    cleaned["second_in_video"] = cleaned["second_in_video"].astype(int)
    cleaned["分类"] = cleaned["分类"].astype(int)
    cleaned["时间戳"] = cleaned["时间戳"].astype("int64")
    cleaned["audio_start_second"] = cleaned["audio_start_second"].astype(int)
    cleaned["audio_duration_seconds"] = cleaned["audio_duration_seconds"].astype(float)
    cleaned["timestamp_inferred"] = cleaned["timestamp_inferred"].astype(bool)
    cleaned["audio_exists"] = cleaned["audio_exists"].astype(bool)
    cleaned["audio_is_silence_fill"] = cleaned["audio_is_silence_fill"].astype(bool)
    cleaned["date"] = pd.to_datetime(cleaned["frame_time"]).dt.strftime("%Y-%m-%d")

    summary["clean_rows"] = int(len(cleaned))
    return cleaned, summary


def assign_split(df: pd.DataFrame, train_dates: list[str], val_dates: list[str], test_dates: list[str]) -> pd.DataFrame:
    split_map = {date: "train" for date in train_dates}
    split_map.update({date: "val" for date in val_dates})
    split_map.update({date: "test" for date in test_dates})
    out = df.copy()
    out["split"] = out["date"].map(split_map)
    missing = out["split"].isna()
    if missing.any():
        missing_dates = sorted(out.loc[missing, "date"].unique().tolist())
        raise SystemExit(f"unassigned dates in cleaned dataset: {missing_dates}")
    return out


def materialize_assets(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, dict]:
    frames_root = output_dir / "frames"
    audio_root = output_dir / "audio_segments"
    videos_root = output_dir / "videos"
    trajectory_root = output_dir / "trajectory_parts"
    date_csv_root = output_dir / "dates"
    for root in [frames_root, audio_root, videos_root, trajectory_root, date_csv_root]:
        root.mkdir(parents=True, exist_ok=True)

    frame_links = {"hardlink": 0, "copy": 0, "existing": 0}
    audio_links = {"hardlink": 0, "copy": 0, "existing": 0}
    video_links = {"symlink": 0, "copy": 0, "existing": 0}

    frame_map: dict[str, str] = {}
    for frame_path_str, group in df.groupby("frame_path", sort=False):
        src = resolve(frame_path_str)
        date = str(group["date"].iloc[0])
        dst = frames_root / date / "aligned_frames" / src.name
        mode = hardlink_or_copy(src, dst)
        frame_links[mode] += 1
        frame_map[str(frame_path_str)] = project_relative(dst)

    audio_map: dict[str, str] = {}
    for audio_path_str, group in df.groupby("audio_path", sort=False):
        src = resolve(audio_path_str)
        split = str(group["split"].iloc[0])
        date = str(group["date"].iloc[0])
        stem = Path(str(group["video_file"].iloc[0])).stem
        dst = audio_root / split / date / stem / src.name
        mode = hardlink_or_copy(src, dst)
        audio_links[mode] += 1
        audio_map[str(audio_path_str)] = project_relative(dst)

    video_map: dict[str, str] = {}
    for video_path_str, group in df.groupby("source_video_path", sort=False):
        src = Path(str(video_path_str))
        date = str(group["date"].iloc[0])
        dst = videos_root / date / src.name
        mode = symlink_or_copy(src, dst)
        video_links[mode] += 1
        video_map[str(video_path_str)] = project_relative(dst)

    out = df.copy()
    out["frame_path"] = out["frame_path"].map(frame_map)
    out["audio_path"] = out["audio_path"].map(audio_map)
    out["video_path"] = out["source_video_path"].map(video_map)
    out["source_video_path"] = out["video_path"]

    return out, {
        "frames": frame_links,
        "audio": audio_links,
        "videos": video_links,
        "unique_frames": int(len(frame_map)),
        "unique_audio_clips": int(len(audio_map)),
        "unique_videos": int(len(video_map)),
    }


def copy_trajectory_parts(part_dir: Path, output_dir: Path) -> int:
    copied = 0
    dst_root = output_dir / "trajectory_parts"
    dst_root.mkdir(parents=True, exist_ok=True)
    for path in sorted(part_dir.glob("*.xlsx")):
        shutil.copy2(path, dst_root / path.name)
        copied += 1
    return copied


def write_split_csvs(df: pd.DataFrame, output_dir: Path) -> dict:
    summaries: dict[str, dict] = {}
    all_columns = [
        "split",
        "frame_path",
        "frame_time",
        "timestamp_source",
        "timestamp_inferred",
        "video_file",
        "video_path",
        "frame_number",
        "second_in_video",
        "经度",
        "纬度",
        "定位时间",
        "速度",
        "方向角",
        "深度",
        "did",
        "分类",
        "时间戳",
        "audio_start_second",
        "audio_duration_seconds",
        "audio_path",
        "audio_exists",
        "audio_is_silence_fill",
    ]

    ordered = df[all_columns + ["date"]].sort_values(["frame_time", "video_file", "second_in_video", "frame_path"]).reset_index(drop=True)
    for split in ["train", "val", "test"]:
        group = ordered[ordered["split"] == split].copy()
        csv_path = output_dir / f"{split}.csv"
        group[all_columns].to_csv(csv_path, index=False, encoding="utf-8-sig")
        summaries[split] = {
            "csv": project_relative(csv_path),
            "rows": int(len(group)),
            "videos": int(group["video_file"].nunique()),
            "dates": sorted(group["date"].unique().tolist()),
            "time_start": str(group["frame_time"].min()) if not group.empty else "",
            "time_end": str(group["frame_time"].max()) if not group.empty else "",
            "class_counts": {str(int(k)): int(v) for k, v in group["分类"].value_counts().sort_index().items()},
        }

    for date, group in ordered.groupby("date", sort=True):
        csv_path = output_dir / "dates" / f"{date}.csv"
        group[all_columns].to_csv(csv_path, index=False, encoding="utf-8-sig")

    all_csv = output_dir / "all.csv"
    ordered[all_columns].to_csv(all_csv, index=False, encoding="utf-8-sig")
    summaries["all"] = {
        "csv": project_relative(all_csv),
        "rows": int(len(ordered)),
        "videos": int(ordered["video_file"].nunique()),
        "dates": sorted(ordered["date"].unique().tolist()),
        "time_start": str(ordered["frame_time"].min()) if not ordered.empty else "",
        "time_end": str(ordered["frame_time"].max()) if not ordered.empty else "",
        "class_counts": {str(int(k)): int(v) for k, v in ordered["分类"].value_counts().sort_index().items()},
    }
    return summaries


def verify_assets(df: pd.DataFrame) -> dict:
    checks = {}
    for column in ["frame_path", "audio_path", "video_path"]:
        paths = [resolve(p) for p in df[column].drop_duplicates().tolist()]
        missing = [str(p) for p in paths if not p.exists()]
        checks[column] = {
            "checked": int(len(paths)),
            "missing": int(len(missing)),
            "missing_examples": missing[:10],
        }
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", default=str(DEFAULT_SOURCE_CSV))
    parser.add_argument("--part-dir", default=str(DEFAULT_PART_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--train-dates", default=",".join(TRAIN_DATES))
    parser.add_argument("--val-dates", default=",".join(VAL_DATES))
    parser.add_argument("--test-dates", default=",".join(TEST_DATES))
    parser.add_argument("--force", action="store_true", help="Remove output-dir before rebuilding")
    args = parser.parse_args()

    source_csv = resolve(args.source_csv)
    part_dir = resolve(args.part_dir)
    output_dir = resolve(args.output_dir)
    train_dates = parse_dates(args.train_dates)
    val_dates = parse_dates(args.val_dates)
    test_dates = parse_dates(args.test_dates)

    if output_dir.exists():
        if not args.force:
            raise SystemExit(f"output dir already exists: {output_dir} (use --force to rebuild)")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = load_source(source_csv)
    cleaned, clean_summary = clean_rows(source)
    assigned = assign_split(cleaned, train_dates, val_dates, test_dates)
    materialized, asset_summary = materialize_assets(assigned, output_dir)
    copied_parts = copy_trajectory_parts(part_dir, output_dir)
    split_summary = write_split_csvs(materialized, output_dir)
    verify_summary = verify_assets(materialized)

    summary = {
        "description": "Cleaned full B_deep_part multimodal dataset with unified video/audio/frame/trajectory assets.",
        "source_csv": str(source_csv),
        "part_dir": str(part_dir),
        "output_dir": str(output_dir),
        "cleaning_rules": {
            "drop_missing_required_fields": True,
            "drop_invalid_labels_outside_0_10": True,
            "drop_rows_with_missing_audio_assets": True,
            "normalize_direction_360_to_0": True,
            "drop_remaining_invalid_direction_rows": True,
            "drop_duplicate_video_second_rows": True,
            "drop_duplicate_frame_path_rows": True,
            "drop_extreme_motion_rows": "gps_speed_mps > max(12.0, speed*1.5 + 2.0) for consecutive 1s rows within the same video",
        },
        "splits": {
            "train_dates": train_dates,
            "val_dates": val_dates,
            "test_dates": test_dates,
        },
        "clean_summary": clean_summary,
        "asset_summary": asset_summary,
        "trajectory_part_files": copied_parts,
        "csv_summary": split_summary,
        "verify_summary": verify_summary,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "rows": split_summary["all"]["rows"],
        "videos": split_summary["all"]["videos"],
        "train_rows": split_summary["train"]["rows"],
        "val_rows": split_summary["val"]["rows"],
        "test_rows": split_summary["test"]["rows"],
        "dropped_extreme_motion_rows": clean_summary["dropped_extreme_motion_rows"],
        "normalized_direction_360_to_0": clean_summary["normalized_direction_360_to_0"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
