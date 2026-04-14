#!/usr/bin/env python3
"""
整理 B 组全量数据集索引，并按日期生成 train/val/test 划分。

输入:
  - /private/data/B_deep           轨迹 Excel
  - /private/data/video/B/<date>   同日期视频目录

输出:
  - data/b_dataset_index/manifest.csv
  - data/b_dataset_index/manifest.json
  - data/b_dataset_index/splits.json
  - data/b_dataset_index/summary.json
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAJECTORY_DIR = Path("/private/data/B_deep")
VIDEO_ROOT = Path("/private/data/video/B")
OUTPUT_DIR = PROJECT_ROOT / "data" / "b_dataset_index"

DATE_SPLITS = {
    "train": [
        "2024-10-18",
        "2024-10-19",
        "2024-10-20",
        "2024-10-22",
        "2024-10-23",
        "2024-10-24",
        "2024-10-25",
    ],
    "val": [
        "2024-10-26",
        "2024-10-27",
    ],
    "test": [
        "2024-10-28",
        "2024-10-29",
    ],
}


def ffprobe_duration(video_path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        text=True,
    ).strip()
    return float(out)


def load_trajectory_stats(xlsx_path: Path) -> Dict[str, object]:
    df = pd.read_excel(xlsx_path)
    time_col = "定位时间" if "定位时间" in df.columns else ("时间" if "时间" in df.columns else None)
    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        traj_start = str(df[time_col].min())
        traj_end = str(df[time_col].max())
    else:
        traj_start = ""
        traj_end = ""

    return {
        "trajectory_rows": int(len(df)),
        "trajectory_start": traj_start,
        "trajectory_end": traj_end,
        "trajectory_columns": list(df.columns),
    }


def collect_manifest() -> List[Dict[str, object]]:
    manifest = []
    for xlsx_path in sorted(TRAJECTORY_DIR.glob("*.xlsx")):
        if xlsx_path.name.startswith("."):
            continue

        parts = xlsx_path.stem.split("_")
        if len(parts) < 2:
            continue
        date = parts[1]
        video_dir = VIDEO_ROOT / date
        videos = sorted(video_dir.glob("*.mp4")) if video_dir.exists() else []

        total_duration = 0.0
        total_size_bytes = 0
        for video_path in videos:
            total_duration += ffprobe_duration(video_path)
            total_size_bytes += video_path.stat().st_size

        traj_stats = load_trajectory_stats(xlsx_path)
        manifest.append(
            {
                "date": date,
                "trajectory_file": str(xlsx_path),
                "video_dir": str(video_dir),
                "video_count": len(videos),
                "video_total_duration_sec": round(total_duration, 2),
                "video_total_duration_hours": round(total_duration / 3600.0, 2),
                "video_total_size_bytes": total_size_bytes,
                "video_total_size_gb": round(total_size_bytes / (1024 ** 3), 3),
                "first_video": videos[0].name if videos else "",
                "last_video": videos[-1].name if videos else "",
                **traj_stats,
            }
        )
    return manifest


def summarize_rows(rows: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "days": len(rows),
        "trajectory_rows": int(sum(int(row["trajectory_rows"]) for row in rows)),
        "video_count": int(sum(int(row["video_count"]) for row in rows)),
        "video_total_duration_sec": round(sum(float(row["video_total_duration_sec"]) for row in rows), 2),
        "video_total_duration_hours": round(sum(float(row["video_total_duration_hours"]) for row in rows), 2),
        "video_total_size_gb": round(sum(float(row["video_total_size_gb"]) for row in rows), 3),
        "dates": [row["date"] for row in rows],
    }


def build_summary(manifest: List[Dict[str, object]]) -> Dict[str, object]:
    split_rows = {
        split: [row for row in manifest if row["date"] in dates]
        for split, dates in DATE_SPLITS.items()
    }
    return {
        "dataset_name": "B_full_dataset",
        "trajectory_dir": str(TRAJECTORY_DIR),
        "video_root": str(VIDEO_ROOT),
        "date_split_strategy": "date-wise chronological split to avoid same-day leakage",
        "totals": summarize_rows(manifest),
        "splits": {split: summarize_rows(rows) for split, rows in split_rows.items()},
    }


def write_outputs(manifest: List[Dict[str, object]], summary: Dict[str, object]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_json = OUTPUT_DIR / "manifest.json"
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    manifest_csv = OUTPUT_DIR / "manifest.csv"
    fieldnames = [
        "date",
        "trajectory_file",
        "trajectory_rows",
        "trajectory_start",
        "trajectory_end",
        "video_dir",
        "video_count",
        "video_total_duration_sec",
        "video_total_duration_hours",
        "video_total_size_bytes",
        "video_total_size_gb",
        "first_video",
        "last_video",
        "trajectory_columns",
    ]
    with open(manifest_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest:
            serializable = dict(row)
            serializable["trajectory_columns"] = json.dumps(row["trajectory_columns"], ensure_ascii=False)
            writer.writerow(serializable)

    splits_json = OUTPUT_DIR / "splits.json"
    with open(splits_json, "w", encoding="utf-8") as f:
        json.dump(DATE_SPLITS, f, ensure_ascii=False, indent=2)

    summary_json = OUTPUT_DIR / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    manifest = collect_manifest()
    summary = build_summary(manifest)
    write_outputs(manifest, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
