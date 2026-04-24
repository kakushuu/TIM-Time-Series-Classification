#!/usr/bin/env python3
"""Build train/val/test CSVs from selected B_deep_part trajectory slices.

The OCR frames have already been aligned in data/b_ocr_dataset. This script
does not rerun OCR. It filters the existing per-date aligned_data.csv files by
timestamps present in /private/data/B_deep_part/*.xlsx and writes chronological
date-wise splits plus an all.csv file containing every selected split row.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PART_DIR = Path("/private/data/B_deep_part")
DEFAULT_OCR_ROOT = PROJECT_ROOT / "data" / "b_ocr_dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "b_deep_part_full_20241018_29"

TRAIN_DATES = ["2024-10-18", "2024-10-19", "2024-10-20", "2024-10-22", "2024-10-23", "2024-10-24", "2024-10-25"]
VAL_DATES = ["2024-10-26", "2024-10-27"]
TEST_DATES = ["2024-10-28", "2024-10-29"]

TRAJ_RENAME = {
    "时间": "定位时间",
    "方向": "方向角",
    "标记": "分类",
}
TRAJ_COLUMNS = ["经度", "纬度", "定位时间", "速度", "方向角", "深度", "did", "分类", "时间戳"]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def parse_dates(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def date_from_part_path(path: Path) -> str | None:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    return match.group(1) if match else None


def load_part_trajectory(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_excel(path).rename(columns=TRAJ_RENAME)
        if "定位时间" not in df.columns:
            raise ValueError(f"missing time column in {path}")
        df["定位时间"] = pd.to_datetime(df["定位时间"], errors="coerce")
        df = df.dropna(subset=["定位时间"])
        df["时间戳"] = (df["定位时间"].astype("int64") // 10**9).astype("int64")
        if "分类" not in df.columns:
            raise ValueError(f"missing label column in {path}")
        df["分类"] = pd.to_numeric(df["分类"], errors="coerce").fillna(-1).astype(int)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=TRAJ_COLUMNS)
    traj = pd.concat(frames, ignore_index=True)
    traj = traj.sort_values("定位时间").drop_duplicates(subset=["时间戳"], keep="first")
    for col in TRAJ_COLUMNS:
        if col not in traj.columns:
            traj[col] = pd.NA
    return traj[TRAJ_COLUMNS].reset_index(drop=True)


def aligned_csv_for_date(ocr_root: Path, date: str) -> Path:
    candidates = [
        ocr_root / "train" / "videos" / date / "aligned_data.csv",
        ocr_root / "val" / "videos" / date / "aligned_data.csv",
        ocr_root / "test" / "videos" / date / "aligned_data.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"no aligned_data.csv found for {date} under {ocr_root}")


def load_filtered_date(ocr_root: Path, date: str, part_paths: list[Path]) -> tuple[pd.DataFrame, dict]:
    trajectory = load_part_trajectory(part_paths)
    aligned_path = aligned_csv_for_date(ocr_root, date)
    aligned = pd.read_csv(aligned_path, encoding="utf-8-sig")
    if aligned.empty or trajectory.empty:
        return pd.DataFrame(columns=aligned.columns), {
            "date": date,
            "part_files": [str(p) for p in part_paths],
            "trajectory_rows": int(len(trajectory)),
            "aligned_source": str(aligned_path),
            "aligned_source_rows": int(len(aligned)),
            "filtered_rows": 0,
            "class_counts": {},
        }

    aligned["定位时间"] = pd.to_datetime(aligned["定位时间"], errors="coerce")
    if "时间戳" not in aligned.columns:
        aligned["时间戳"] = aligned["定位时间"].astype("int64") // 10**9
    aligned["时间戳"] = pd.to_numeric(aligned["时间戳"], errors="coerce").fillna(-1).astype("int64")

    selected = trajectory.set_index("时间戳")
    filtered = aligned[aligned["时间戳"].isin(selected.index)].copy()
    if not filtered.empty:
        overlay = selected.reindex(filtered["时间戳"]).reset_index(drop=True)
        for col in TRAJ_COLUMNS:
            if col in overlay.columns:
                filtered[col] = overlay[col].to_numpy()
        filtered["定位时间"] = pd.to_datetime(filtered["定位时间"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        filtered["分类"] = pd.to_numeric(filtered["分类"], errors="coerce").fillna(-1).astype(int)
        sort_cols = [c for c in ["frame_time", "video_file", "second_in_video", "frame_path"] if c in filtered.columns]
        filtered = filtered.sort_values(sort_cols).reset_index(drop=True)

    class_counts = {
        str(int(k)): int(v)
        for k, v in filtered["分类"].value_counts().sort_index().items()
    } if "分类" in filtered.columns else {}
    return filtered, {
        "date": date,
        "part_files": [str(p) for p in part_paths],
        "trajectory_rows": int(len(trajectory)),
        "aligned_source": str(aligned_path),
        "aligned_source_rows": int(len(aligned)),
        "filtered_rows": int(len(filtered)),
        "class_counts": class_counts,
        "time_start": str(filtered["frame_time"].min()) if not filtered.empty and "frame_time" in filtered.columns else "",
        "time_end": str(filtered["frame_time"].max()) if not filtered.empty and "frame_time" in filtered.columns else "",
    }


def build_split(name: str, dates: list[str], ocr_root: Path, parts_by_date: dict[str, list[Path]], output_dir: Path) -> dict:
    frames = []
    date_summaries = []
    date_dir = output_dir / "dates"
    date_dir.mkdir(parents=True, exist_ok=True)
    for date in dates:
        part_paths = parts_by_date.get(date, [])
        if not part_paths:
            raise FileNotFoundError(f"no B_deep_part xlsx files found for {date}")
        filtered, summary = load_filtered_date(ocr_root, date, part_paths)
        filtered.to_csv(date_dir / f"{date}.csv", index=False, encoding="utf-8-sig")
        date_summaries.append(summary)
        if not filtered.empty:
            frames.append(filtered)
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not merged.empty:
        sort_cols = [c for c in ["frame_time", "video_file", "second_in_video", "frame_path"] if c in merged.columns]
        merged = merged.sort_values(sort_cols).reset_index(drop=True)
    split_csv = output_dir / f"{name}.csv"
    merged.to_csv(split_csv, index=False, encoding="utf-8-sig")
    return {
        "csv": str(split_csv.relative_to(PROJECT_ROOT) if split_csv.is_relative_to(PROJECT_ROOT) else split_csv),
        "dates": dates,
        "rows": int(len(merged)),
        "videos": int(merged["video_file"].nunique()) if not merged.empty and "video_file" in merged.columns else 0,
        "time_start": str(merged["frame_time"].min()) if not merged.empty and "frame_time" in merged.columns else "",
        "time_end": str(merged["frame_time"].max()) if not merged.empty and "frame_time" in merged.columns else "",
        "class_counts": {
            str(int(k)): int(v)
            for k, v in merged["分类"].value_counts().sort_index().items()
        } if not merged.empty and "分类" in merged.columns else {},
        "date_summaries": date_summaries,
    }


def write_all_csv(output_dir: Path, split_summaries: dict[str, dict]) -> dict:
    frames = []
    for split in ["train", "val", "test"]:
        csv_path = resolve(split_summaries[split]["csv"])
        if csv_path.exists():
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            if not df.empty:
                df.insert(0, "split", split)
                frames.append(df)
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not merged.empty:
        sort_cols = [c for c in ["frame_time", "video_file", "second_in_video", "frame_path"] if c in merged.columns]
        merged = merged.sort_values(sort_cols).reset_index(drop=True)
    all_csv = output_dir / "all.csv"
    merged.to_csv(all_csv, index=False, encoding="utf-8-sig")
    return {
        "csv": str(all_csv.relative_to(PROJECT_ROOT) if all_csv.is_relative_to(PROJECT_ROOT) else all_csv),
        "rows": int(len(merged)),
        "videos": int(merged["video_file"].nunique()) if not merged.empty and "video_file" in merged.columns else 0,
        "time_start": str(merged["frame_time"].min()) if not merged.empty and "frame_time" in merged.columns else "",
        "time_end": str(merged["frame_time"].max()) if not merged.empty and "frame_time" in merged.columns else "",
        "class_counts": {
            str(int(k)): int(v)
            for k, v in merged["分类"].value_counts().sort_index().items()
        } if not merged.empty and "分类" in merged.columns else {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part-dir", default=str(DEFAULT_PART_DIR))
    parser.add_argument("--ocr-root", default=str(DEFAULT_OCR_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--train-dates", default=",".join(TRAIN_DATES))
    parser.add_argument("--val-dates", default=",".join(VAL_DATES))
    parser.add_argument("--test-dates", default=",".join(TEST_DATES))
    args = parser.parse_args()

    part_dir = resolve(args.part_dir)
    ocr_root = resolve(args.ocr_root)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parts_by_date: dict[str, list[Path]] = {}
    for path in sorted(part_dir.glob("*.xlsx")):
        date = date_from_part_path(path)
        if date:
            parts_by_date.setdefault(date, []).append(path)

    splits = {
        "train": parse_dates(args.train_dates),
        "val": parse_dates(args.val_dates),
        "test": parse_dates(args.test_dates),
    }
    summary = {
        "description": "B_deep_part timestamp-filtered OCR splits generated from all selected /private/data/B_deep_part rows covered by the configured dates.",
        "part_dir": str(part_dir),
        "ocr_root": str(ocr_root),
        "splits": {},
    }
    for split, dates in splits.items():
        summary["splits"][split] = build_split(split, dates, ocr_root, parts_by_date, output_dir)
    summary["all"] = write_all_csv(output_dir, summary["splits"])

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
