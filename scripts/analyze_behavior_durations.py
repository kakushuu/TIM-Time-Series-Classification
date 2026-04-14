#!/usr/bin/env python3
"""
Summarize contiguous behavior durations and derive adaptive sampling hints.

Each segment is a same-label run inside one video. A run is split when the label
changes or when second_in_video has a gap larger than --max-gap.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-agri-mbt")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = [
    "Reverse empty harvesting",
    "Straight empty harvesting",
    "Turning empty harvesting",
    "Full-load harvesting",
    "Reverse transfer",
    "Straight transfer",
    "Turning transfer",
    "Engine-off waiting",
    "Idling waiting",
    "Unloading",
    "Road driving",
]


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze behavior duration distribution")
    parser.add_argument("--input-csv", default="data/b_ocr_dataset/train/aligned_data.csv")
    parser.add_argument("--output-dir", default="experiments/behavior_duration_analysis")
    parser.add_argument("--max-gap", type=int, default=1, help="Split a segment if second gap is larger than this")
    parser.add_argument("--min-window", type=int, default=16)
    parser.add_argument("--max-window", type=int, default=512)
    parser.add_argument("--context-scale", type=float, default=2.0)
    parser.add_argument("--min-stride", type=int, default=1)
    parser.add_argument("--max-stride", type=int, default=20)
    parser.add_argument("--stride-ratio", type=float, default=0.25)
    return parser.parse_args()


def segment_video(group: pd.DataFrame, max_gap: int) -> list[dict]:
    group = group.sort_values(["second_in_video", "frame_time"]).reset_index(drop=True)
    rows = []
    if group.empty:
        return rows

    start = 0
    prev_label = int(group.loc[0, "分类"])
    prev_second = int(group.loc[0, "second_in_video"])
    for i in range(1, len(group)):
        label = int(group.loc[i, "分类"])
        second = int(group.loc[i, "second_in_video"])
        if label != prev_label or second - prev_second > max_gap:
            rows.append(make_segment(group, start, i - 1))
            start = i
        prev_label = label
        prev_second = second
    rows.append(make_segment(group, start, len(group) - 1))
    return rows


def make_segment(group: pd.DataFrame, start: int, end: int) -> dict:
    first_second = int(group.loc[start, "second_in_video"])
    last_second = int(group.loc[end, "second_in_video"])
    class_id = int(group.loc[start, "分类"])
    return {
        "video_file": group.loc[start, "video_file"],
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else str(class_id),
        "start_second": first_second,
        "end_second": last_second,
        "start_time": str(group.loc[start, "frame_time"]),
        "end_time": str(group.loc[end, "frame_time"]),
        "rows": int(end - start + 1),
        "duration_seconds": int(max(last_second - first_second + 1, end - start + 1)),
    }


def summarize(segments: pd.DataFrame, args) -> pd.DataFrame:
    records = []
    for class_id in range(len(CLASS_NAMES)):
        sub = segments[segments["class_id"] == class_id]
        durations = sub["duration_seconds"].to_numpy(dtype=np.float32)
        if len(durations) == 0:
            stats = {
                "segment_count": 0,
                "total_seconds": 0,
                "min": 0,
                "p10": 0,
                "p25": 0,
                "median": 0,
                "mean": 0,
                "p75": 0,
                "p90": 0,
                "p95": 0,
                "max": 0,
            }
        else:
            stats = {
                "segment_count": int(len(durations)),
                "total_seconds": int(durations.sum()),
                "min": float(np.min(durations)),
                "p10": float(np.percentile(durations, 10)),
                "p25": float(np.percentile(durations, 25)),
                "median": float(np.percentile(durations, 50)),
                "mean": float(np.mean(durations)),
                "p75": float(np.percentile(durations, 75)),
                "p90": float(np.percentile(durations, 90)),
                "p95": float(np.percentile(durations, 95)),
                "max": float(np.max(durations)),
            }

        base_window = max(stats["p75"] * args.context_scale, stats["median"], 1)
        window = int(np.clip(np.ceil(base_window), args.min_window, args.max_window))
        stride = int(np.clip(np.ceil(max(stats["median"], 1) * args.stride_ratio), args.min_stride, args.max_stride))
        records.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            **stats,
            "recommended_window": window,
            "recommended_stride": stride,
        })
    return pd.DataFrame(records)


def save_plot(segments: pd.DataFrame, summary: pd.DataFrame, output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5), dpi=160)
    data = [
        segments.loc[segments["class_id"] == class_id, "duration_seconds"].to_numpy()
        for class_id in range(len(CLASS_NAMES))
    ]
    ax.boxplot(data, tick_labels=[str(i) for i in range(len(CLASS_NAMES))], showfliers=False)
    ax.set_title("Behavior Duration Distribution")
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Duration (seconds)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "duration_distribution.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=160)
    x = np.arange(len(summary))
    ax.bar(x - 0.2, summary["recommended_window"], width=0.4, label="Window")
    ax.bar(x + 0.2, summary["recommended_stride"], width=0.4, label="Stride")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["class_id"].astype(str))
    ax.set_title("Recommended Adaptive Sampling")
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Seconds")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "recommended_sampling.png")
    plt.close(fig)


def main():
    args = parse_args()
    input_csv = resolve(args.input_csv)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    required = {"video_file", "frame_time", "second_in_video", "分类"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df["frame_time"] = pd.to_datetime(df["frame_time"])
    df["second_in_video"] = df["second_in_video"].astype(int)
    df["分类"] = df["分类"].astype(int)
    df = df.sort_values(["video_file", "second_in_video", "frame_time"]).reset_index(drop=True)

    segment_rows = []
    for _, group in df.groupby("video_file", sort=False):
        segment_rows.extend(segment_video(group, args.max_gap))
    segments = pd.DataFrame(segment_rows)
    summary = summarize(segments, args)

    segments.to_csv(output_dir / "segments.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "duration_summary.csv", index=False, encoding="utf-8-sig")
    config = {
        str(int(row.class_id)): {
            "window": int(row.recommended_window),
            "stride": int(row.recommended_stride),
            "median_duration": float(row.median),
            "p75_duration": float(row.p75),
            "segment_count": int(row.segment_count),
        }
        for row in summary.itertuples()
    }
    (output_dir / "duration_sampling_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_plot(segments, summary, output_dir)

    print(f"Segments: {len(segments)}")
    print(summary[["class_id", "segment_count", "median", "p75", "recommended_window", "recommended_stride"]].to_string(index=False))
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
