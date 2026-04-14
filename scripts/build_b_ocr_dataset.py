#!/usr/bin/env python3
"""
按 split/date 批量运行 OCR 对齐流程，生成最终训练数据集目录。

目录结构:
  data/b_ocr_dataset/
    train/
      videos/<date>/
      trajectories/<date>/
      aligned_data.csv
      aligned_data.json
      summary.json
    val/
      ...
    test/
      ...
    manifest.json
    splits.json
    summary.json

说明:
- 不复制原始大文件。原始视频目录和轨迹 xlsx 使用软链接接入数据集目录。
- OCR 提取帧、对齐帧和 split 级聚合 CSV 会写入目标数据集目录。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from align_complete_pipeline import (
    FrameDeduplicator,
    FrameTimeRecognizer,
    TrajectoryAligner,
    VideoFrameExtractor,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "b_dataset_index"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "b_ocr_dataset"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        dst.unlink()
    dst.symlink_to(src)


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def process_date(
    row: Dict[str, object],
    split: str,
    output_root: Path,
    time_tolerance: int,
    quiet: bool,
    skip_existing: bool,
) -> Dict[str, object]:
    date = str(row["date"])
    video_dir = Path(str(row["video_dir"]))
    trajectory_file = Path(str(row["trajectory_file"]))

    split_root = output_root / split
    date_video_root = split_root / "videos" / date
    date_traj_root = split_root / "trajectories" / date

    ensure_clean_dir(date_video_root)
    ensure_clean_dir(date_traj_root)

    ensure_symlink(video_dir, date_video_root / "raw")
    ensure_symlink(trajectory_file, date_traj_root / trajectory_file.name)

    output_csv = date_video_root / "aligned_data.csv"
    output_json = date_video_root / "aligned_data.json"
    output_stats = date_video_root / "alignment_stats.json"
    final_frames_dir = date_video_root / "aligned_frames"

    if skip_existing and output_csv.exists() and final_frames_dir.exists():
        existing = pd.read_csv(output_csv, encoding="utf-8-sig")
        return {
            "date": date,
            "split": split,
            "aligned_rows": int(len(existing)),
            "aligned_frame_count": int(len(list(final_frames_dir.glob("*.jpg")))),
            "output_dir": str(date_video_root),
            "skipped": True,
        }

    extractor = VideoFrameExtractor(
        video_dir=str(video_dir),
        output_dir=str(date_video_root),
        verbose=not quiet,
    )
    extractor.extract_frames_from_all_videos()

    frames_dir = date_video_root / "extracted_frames"
    recognizer = FrameTimeRecognizer(
        frames_dir=str(frames_dir),
        verbose=not quiet,
    )
    recognizer.recognize_and_rename_all()

    deduplicator = FrameDeduplicator(
        frames_dir=str(frames_dir),
        verbose=not quiet,
    )
    deduplicator.remove_duplicates()

    aligner = TrajectoryAligner(
        trajectory_path=str(trajectory_file),
        frames_dir=str(frames_dir),
        output_dir=str(date_video_root),
        video_dir=str(video_dir),
        time_tolerance=time_tolerance,
        verbose=not quiet,
    )
    aligned_df = aligner.align_frames_with_trajectory()
    if len(aligned_df) > 0:
        aligner.save_results(aligned_df)
    else:
        aligned_df = pd.DataFrame()
        aligned_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        output_json.write_text("[]\n", encoding="utf-8")
        output_stats.write_text(
            json.dumps(
                {
                    "total_aligned_frames": 0,
                    "output_directory": str(date_video_root),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    for file_name in ["aligned_data.csv", "aligned_data.json", "alignment_stats.json"]:
        src = date_video_root / file_name
        ensure_symlink(src, date_traj_root / file_name)

    return {
        "date": date,
        "split": split,
        "aligned_rows": int(len(aligned_df)),
        "aligned_frame_count": int(len(list(final_frames_dir.glob("*.jpg")))) if final_frames_dir.exists() else 0,
        "output_dir": str(date_video_root),
        "skipped": False,
    }


def aggregate_split(split: str, split_root: Path, processed_dates: List[Dict[str, object]]) -> Dict[str, object]:
    frames = []
    for date_info in processed_dates:
        csv_path = split_root / "videos" / date_info["date"] / "aligned_data.csv"
        if csv_path.exists() and csv_path.stat().st_size > 0:
            try:
                df = pd.read_csv(csv_path, encoding="utf-8-sig")
            except pd.errors.EmptyDataError:
                continue
            if len(df) > 0:
                frames.append(df)

    if frames:
        merged = pd.concat(frames, ignore_index=True)
        sort_cols = [col for col in ["frame_time", "video_file", "second_in_video"] if col in merged.columns]
        if sort_cols:
            merged = merged.sort_values(sort_cols).reset_index(drop=True)
    else:
        merged = pd.DataFrame()

    split_csv = split_root / "aligned_data.csv"
    split_json = split_root / "aligned_data.json"
    split_summary = split_root / "summary.json"

    merged.to_csv(split_csv, index=False, encoding="utf-8-sig")
    merged.to_json(split_json, orient="records", force_ascii=False, indent=2)

    summary = {
        "split": split,
        "dates": [item["date"] for item in processed_dates],
        "date_count": len(processed_dates),
        "aligned_rows": int(len(merged)),
        "aligned_frame_count": int(sum(item["aligned_frame_count"] for item in processed_dates)),
        "processed_dates": processed_dates,
    }
    split_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_dataset(
    manifest: List[Dict[str, object]],
    splits: Dict[str, List[str]],
    output_root: Path,
    time_tolerance: int,
    quiet: bool,
    skip_existing: bool,
    only_splits: List[str] | None,
    only_dates: List[str] | None,
) -> Dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    rows_by_date = {str(row["date"]): row for row in manifest}

    root_manifest = output_root / "manifest.json"
    root_splits = output_root / "splits.json"
    root_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    root_splits.write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")

    dataset_summary = {
        "dataset_root": str(output_root),
        "time_tolerance": time_tolerance,
        "splits": {},
    }

    for split, dates in splits.items():
        if only_splits and split not in only_splits:
            continue

        split_root = output_root / split
        ensure_clean_dir(split_root / "videos")
        ensure_clean_dir(split_root / "trajectories")

        selected_dates = [date for date in dates if not only_dates or date in only_dates]
        processed_dates = []
        for date in tqdm(
            selected_dates,
            desc=f"{split} dates",
            unit="date",
            dynamic_ncols=True,
            disable=quiet,
        ):
            row = rows_by_date.get(date)
            if row is None:
                continue
            processed_dates.append(
                process_date(
                    row=row,
                    split=split,
                    output_root=output_root,
                    time_tolerance=time_tolerance,
                    quiet=quiet,
                    skip_existing=skip_existing,
                )
            )

        dataset_summary["splits"][split] = aggregate_split(split, split_root, processed_dates)

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(dataset_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return dataset_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 B 组 OCR 最终训练数据集")
    parser.add_argument(
        "--index-dir",
        default=str(INDEX_DIR),
        help="数据索引目录，需包含 manifest.json 和 splits.json",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="最终数据集输出目录",
    )
    parser.add_argument(
        "--time-tolerance",
        type=int,
        default=2,
        help="轨迹对齐时间容差（秒）",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="只处理指定 split，例如 train val",
    )
    parser.add_argument(
        "--dates",
        nargs="*",
        default=None,
        help="只处理指定日期，例如 2024-10-18 2024-10-19",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="若目标日期已存在 aligned_data.csv，则跳过该日期",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="减少处理日志",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_dir = Path(args.index_dir)
    output_root = Path(args.output_root)

    manifest = load_json(index_dir / "manifest.json")
    splits = load_json(index_dir / "splits.json")

    summary = build_dataset(
        manifest=manifest,
        splits=splits,
        output_root=output_root,
        time_tolerance=args.time_tolerance,
        quiet=args.quiet,
        skip_existing=args.skip_existing,
        only_splits=args.splits,
        only_dates=args.dates,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
