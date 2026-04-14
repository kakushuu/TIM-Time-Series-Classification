#!/usr/bin/env python3
"""Plot B_deep spatial label distributions by date with optional predictions."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-deep-dir", default="/private/data/B_deep")
    parser.add_argument("--predictions", action="append", default=[], help="Optional predictions.csv path; repeatable")
    parser.add_argument("--exclude-date", action="append", default=["2024-10-18"])
    parser.add_argument("--output", default="experiments/agri_image_autoresearch/b_deep_spatial_by_date.png")
    parser.add_argument("--max-points-per-panel", type=int, default=5000)
    return parser.parse_args()


def date_from_name(path: Path) -> str:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    return match.group(1) if match else ""


def load_b_deep(root: Path, excluded: set[str]) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("*.xlsx")):
        date = date_from_name(path)
        if not date or date in excluded:
            continue
        df = pd.read_excel(path)
        needed = {"经度", "纬度", "标记"}
        if not needed.issubset(df.columns):
            continue
        df = df[["经度", "纬度", "标记"]].copy()
        df["date"] = date
        df["kind"] = "B_deep true"
        df["class_id"] = df["标记"].astype(int)
        frames.append(df.drop(columns=["标记"]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_predictions(paths: list[str], excluded: set[str]) -> pd.DataFrame:
    frames = []
    for item in paths:
        path = Path(item)
        df = pd.read_csv(path, encoding="utf-8-sig")
        needed = {"frame_time", "经度", "纬度", "y_true", "y_pred"}
        if not needed.issubset(df.columns):
            continue
        df["date"] = pd.to_datetime(df["frame_time"]).dt.strftime("%Y-%m-%d")
        df = df[~df["date"].isin(excluded)].copy()
        for col, kind in [("y_true", "Eval true"), ("y_pred", "Eval pred")]:
            part = df[["经度", "纬度", "date", col]].copy()
            part["kind"] = kind
            part["class_id"] = part[col].astype(int)
            frames.append(part.drop(columns=[col]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def downsample(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if max_points <= 0:
        return df
    parts = []
    for _, group in df.groupby(["date", "kind"], sort=False):
        if len(group) > max_points:
            parts.append(group.sample(max_points, random_state=42))
        else:
            parts.append(group)
    return pd.concat(parts, ignore_index=True) if parts else df


def main() -> None:
    args = parse_args()
    excluded = set(args.exclude_date)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.concat([
        load_b_deep(Path(args.b_deep_dir), excluded),
        load_predictions(args.predictions, excluded),
    ], ignore_index=True)
    if df.empty:
        raise SystemExit("No rows to plot after date filtering.")
    df = downsample(df.dropna(subset=["经度", "纬度", "class_id"]), args.max_points_per_panel)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")
    dates = sorted(df["date"].unique())
    kinds = sorted(df["kind"].unique())
    fig, axes = plt.subplots(len(dates), len(kinds), figsize=(6 * len(kinds), 4.8 * len(dates)), squeeze=False)
    palette = sns.color_palette("tab20", 11)
    for r, date in enumerate(dates):
        for c, kind in enumerate(kinds):
            ax = axes[r][c]
            part = df[(df["date"] == date) & (df["kind"] == kind)]
            if part.empty:
                ax.axis("off")
                continue
            sns.scatterplot(
                data=part,
                x="经度",
                y="纬度",
                hue="class_id",
                palette=palette,
                s=9,
                linewidth=0,
                alpha=0.72,
                ax=ax,
                legend=(r == 0 and c == len(kinds) - 1),
            )
            ax.set_title(f"{date} | {kind}")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
