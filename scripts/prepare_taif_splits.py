#!/usr/bin/env python3
"""
Prepare explicit train/val/test CSV splits for TAIF training.

The split is temporal within each video_file. Boundary rows are dropped so
sliding windows from adjacent splits cannot share frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare TAIF train/val/test CSV splits")
    parser.add_argument(
        "--input-csv",
        default="data/b_ocr_dataset/train/videos/2024-10-18/aligned_data.csv",
        help="Aligned CSV to split",
    )
    parser.add_argument(
        "--output-dir",
        default="data/taif_20241018_split",
        help="Directory for train.csv, val.csv, test.csv and summary.json",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Training window size; split boundaries drop window_size-1 rows",
    )
    parser.add_argument(
        "--min-video-rows",
        type=int,
        default=10,
        help="Skip videos with fewer rows than this",
    )
    return parser.parse_args()


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "rows": 0,
            "videos": 0,
            "time_start": None,
            "time_end": None,
            "class_counts": {},
        }
    return {
        "rows": int(len(df)),
        "videos": int(df["video_file"].nunique()),
        "time_start": str(df["frame_time"].min()),
        "time_end": str(df["frame_time"].max()),
        "class_counts": {
            str(int(k)): int(v)
            for k, v in df["分类"].value_counts().sort_index().items()
        },
    }


def main() -> None:
    args = parse_args()
    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6

    input_csv = resolve_path(args.input_csv)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    required = {"frame_path", "frame_time", "video_file", "second_in_video", "分类"}
    missing = sorted(required - set(df.columns))
    assert not missing, f"Input CSV missing columns: {missing}"

    df["frame_time"] = pd.to_datetime(df["frame_time"])
    df["second_in_video"] = df["second_in_video"].astype(int)
    df = df.sort_values(["video_file", "second_in_video", "frame_time"]).reset_index(drop=True)

    gap = max(args.window_size - 1, 0)
    split_parts = {"train": [], "val": [], "test": []}
    skipped_videos = []
    per_video = []

    for video_file, group in df.groupby("video_file", sort=False):
        group = group.sort_values(["second_in_video", "frame_time"]).reset_index(drop=True)
        n = len(group)
        if n < args.min_video_rows:
            skipped_videos.append({"video_file": video_file, "rows": int(n)})
            continue

        train_end = int(n * args.train_ratio)
        val_len = int(n * args.val_ratio)
        val_start = min(train_end + gap, n)
        val_end = min(val_start + val_len, n)
        test_start = min(val_end + gap, n)

        train = group.iloc[:train_end].copy()
        val = group.iloc[val_start:val_end].copy()
        test = group.iloc[test_start:].copy()

        if not train.empty:
            split_parts["train"].append(train)
        if not val.empty:
            split_parts["val"].append(val)
        if not test.empty:
            split_parts["test"].append(test)

        per_video.append(
            {
                "video_file": video_file,
                "rows": int(n),
                "train_rows": int(len(train)),
                "val_rows": int(len(val)),
                "test_rows": int(len(test)),
                "dropped_boundary_rows": int(n - len(train) - len(val) - len(test)),
            }
        )

    outputs = {}
    for split, parts in split_parts.items():
        out_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=df.columns)
        out_df = out_df.sort_values(["frame_time", "video_file", "second_in_video"]).reset_index(drop=True)
        out_df["frame_time"] = out_df["frame_time"].astype(str)
        output_csv = output_dir / f"{split}.csv"
        out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        outputs[split] = {
            "csv": str(output_csv.relative_to(PROJECT_ROOT)),
            **summarize(out_df),
        }

    summary = {
        "input_csv": str(input_csv.relative_to(PROJECT_ROOT)),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "window_size": args.window_size,
        "gap_rows_per_boundary": gap,
        "splits": outputs,
        "per_video": per_video,
        "skipped_videos": skipped_videos,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
