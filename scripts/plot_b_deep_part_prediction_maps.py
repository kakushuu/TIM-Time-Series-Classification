#!/usr/bin/env python3
"""Plot per-part 2024-10-27 trajectory prediction diagnostics.

The default training artifacts plot all fields together, which hides the
dominant failure modes. This script writes one set of figures per field part / source Excel:

* local_meter_error_map: local meter coordinates, wrong points colored by error pair
* spatial_errors: correct/wrong scatter in local meter coordinates
* true_vs_pred_timeline: true/predicted class strips over elapsed time
* confusion: per-part row-normalized confusion matrix
* true_pred_class_distribution: per-part true/predicted class distribution comparison
* error_pair_lollipop: top true->predicted error-pair count comparison
"""

from __future__ import annotations

import argparse
import os
import re
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
CLASS_SHORT = ["R-EH", "S-EH", "T-EH", "Full", "R-Tr", "S-Tr", "T-Tr", "Off", "Idle", "Unload", "Road"]
OTHER_ERROR_COLOR = "#6b1f1f"
PAIR_COLORS = [
    "#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b",
    "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#aec7e8", "#ffbb78",
]
PAPER_PALETTE = {
    "accuracy": "#3b7f5f",
    "error_rate": "#b85c5c",
    "true": "#4c78a8",
    "predicted": "#f58518",
    "grid": "#d9d9d9",
    "text": "#222222",
}


def apply_paper_style() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "axes.edgecolor": "#333333",
        "axes.labelcolor": PAPER_PALETTE["text"],
        "axes.titlecolor": PAPER_PALETTE["text"],
        "axes.titleweight": "semibold",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "grid.color": PAPER_PALETTE["grid"],
        "grid.linewidth": 0.6,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
    })


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def part_number(path: Path) -> int:
    match = re.search(r"_part(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def load_part_ranges(part_dir: Path, date: str) -> list[dict]:
    ranges = []
    for path in sorted(part_dir.glob(f"*{date}*_part*.xlsx"), key=part_number):
        df = pd.read_excel(path)
        time_col = "时间" if "时间" in df.columns else "定位时间"
        df = df.copy()
        df["_plot_time"] = pd.to_datetime(df[time_col], errors="coerce")
        for col in ["经度", "纬度"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["_plot_time", "经度", "纬度"]).sort_values("_plot_time").reset_index(drop=True)
        times = df["_plot_time"]
        if times.empty:
            continue
        ranges.append({
            "name": path.stem,
            "part": part_number(path),
            "path": path,
            "start": times.min(),
            "end": times.max(),
            "trajectory": df[["_plot_time", "经度", "纬度"]].copy(),
        })
    return ranges


def subset_for_part(df: pd.DataFrame, part: dict) -> pd.DataFrame:
    sub = df[(df["frame_time"] >= part["start"]) & (df["frame_time"] <= part["end"])].copy()
    if sub.empty:
        return sub
    sub = sub.sort_values("frame_time").reset_index(drop=True)
    sub["correct"] = sub["correct"].astype(bool)
    sub["y_true"] = sub["y_true"].astype(int)
    sub["y_pred"] = sub["y_pred"].astype(int)
    sub["elapsed_min"] = (sub["frame_time"] - sub["frame_time"].iloc[0]).dt.total_seconds() / 60.0
    return sub


def add_class_legend(ax) -> None:
    from matplotlib.lines import Line2D

    cmap = ax.figure._class_cmap
    handles = [
        Line2D([0], [0], marker="s", color="w", label=f"{i}: {CLASS_SHORT[i]}",
               markerfacecolor=cmap(i), markersize=7)
        for i in range(11)
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.95)


def local_xy_meters(
    sub: pd.DataFrame,
    origin: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    lon = pd.to_numeric(sub["经度"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(sub["纬度"], errors="coerce").to_numpy(dtype=float)
    if origin is None:
        lon0 = float(np.nanmin(lon))
        lat0 = float(np.nanmin(lat))
    else:
        lon0, lat0 = origin
    x = (lon - lon0) * 111_320.0 * np.cos(np.deg2rad(lat0))
    y = (lat - lat0) * 110_540.0
    return x, y, (lon0, lat0)


def part_background_xy(part: dict) -> tuple[np.ndarray, np.ndarray, tuple[float, float]] | None:
    raw = part.get("trajectory")
    if raw is None or raw.empty:
        return None
    return local_xy_meters(raw)


def draw_part_background(ax, part: dict, origin: tuple[float, float] | None = None) -> tuple[float, float] | None:
    raw = part.get("trajectory")
    if raw is None or raw.empty:
        return origin
    x, y, bg_origin = local_xy_meters(raw, origin=origin)
    ax.plot(x, y, color="#bdbdbd", linewidth=0.7, alpha=0.55, zorder=1, label="Excel full trajectory")
    ax.scatter(x, y, s=5, color="#bdbdbd", alpha=0.18, linewidths=0, zorder=1)
    return bg_origin


def plot_local_error_map(sub: pd.DataFrame, part: dict, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    background = part_background_xy(part)
    if background is None:
        x, y, origin = local_xy_meters(sub)
    else:
        _, _, origin = background
        x, y, _ = local_xy_meters(sub, origin=origin)
    wrong = sub[~sub["correct"]].copy()
    if not wrong.empty:
        wx, wy, _ = local_xy_meters(wrong, origin=origin)
    else:
        wx, wy = np.array([]), np.array([])

    acc = sub["correct"].mean()
    fig, ax = plt.subplots(figsize=(8.5, 8), dpi=180)
    draw_part_background(ax, part, origin=origin)
    ax.scatter(x, y, s=7, color="#b7c8b0", alpha=0.34, linewidths=0, label="predicted samples", zorder=2)

    if wrong.empty:
        ax.set_title(f"Trajectory {part['part']} local error map (accuracy={acc:.1%}, n={len(sub)}, no errors)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.22)
        ax.set_xlabel("Local east coordinate (m)")
        ax.set_ylabel("Local north coordinate (m)")
        fig.tight_layout()
        fig.savefig(output_dir / f"part{part['part']:02d}_local_meter_error_map.png")
        plt.close(fig)
        return

    pairs = wrong.groupby(["y_true", "y_pred"]).size().sort_values(ascending=False).head(8)
    handles = []
    for idx, ((y_true, y_pred), count) in enumerate(pairs.items()):
        mask = (wrong["y_true"].to_numpy() == y_true) & (wrong["y_pred"].to_numpy() == y_pred)
        color = PAIR_COLORS[idx % len(PAIR_COLORS)]
        ax.scatter(wx[mask], wy[mask], s=18, marker="o", color=color, alpha=0.88, edgecolors="white", linewidths=0.25, zorder=3)
        handles.append(Line2D([0], [0], marker="o", color="w", label=f"{y_true}->{y_pred} {CLASS_SHORT[y_true]}->{CLASS_SHORT[y_pred]} n={int(count)}", markerfacecolor=color, markeredgecolor="white", linestyle="None", markersize=6))
    other_pairs = set(pairs.index)
    other_mask = np.array([(t, p) not in other_pairs for t, p in zip(wrong["y_true"], wrong["y_pred"])])
    if other_mask.any():
        ax.scatter(wx[other_mask], wy[other_mask], s=10, marker="o", color=OTHER_ERROR_COLOR, alpha=0.45, linewidths=0, zorder=2)
        handles.append(Line2D([0], [0], marker="o", color="w", label=f"other n={int(other_mask.sum())}", markerfacecolor=OTHER_ERROR_COLOR, linestyle="None", markersize=6, alpha=0.65))
    ax.legend(handles=handles, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.95)
    ax.set_title(f"Trajectory {part['part']} local error map (accuracy={acc:.1%}, n={len(sub)}, errors={len(wrong)})")
    ax.set_xlabel("Local east coordinate (m)")
    ax.set_ylabel("Local north coordinate (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_dir / f"part{part['part']:02d}_local_meter_error_map.png")
    plt.close(fig)


def plot_spatial_errors_by_part(sub: pd.DataFrame, part: dict, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_paper_style()
    background = part_background_xy(part)
    if background is None:
        x, y, origin = local_xy_meters(sub)
    else:
        _, _, origin = background
        x, y, _ = local_xy_meters(sub, origin=origin)
    correct = sub["correct"].astype(bool).to_numpy()
    acc = float(correct.mean()) if len(correct) else 0.0

    fig, ax = plt.subplots(figsize=(8.2, 7.2), dpi=180)
    draw_part_background(ax, part, origin=origin)
    if correct.any():
        ax.scatter(
            x[correct],
            y[correct],
            c="#6aa36f",
            s=8,
            alpha=0.42,
            label="Correct",
            linewidths=0,
            zorder=2,
        )
    if (~correct).any():
        ax.scatter(
            x[~correct],
            y[~correct],
            c="#d84a4a",
            s=18,
            alpha=0.82,
            marker="x",
            label="Wrong",
            linewidths=0.8,
            zorder=3,
        )
    ax.set_title(f"Trajectory {part['part']} spatial prediction errors (accuracy={acc:.1%})")
    ax.set_xlabel("Local east coordinate (m)")
    ax.set_ylabel("Local north coordinate (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper left", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(output_dir / f"part{part['part']:02d}_spatial_errors.png")
    plt.close(fig)


def plot_timeline(sub: pd.DataFrame, part: dict, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    colors = list(plt.get_cmap("tab20").colors[:11])
    cmap = ListedColormap(colors)
    x = sub["elapsed_min"].to_numpy(dtype=float)
    true_values = sub["y_true"].to_numpy(dtype=int)
    pred_values = sub["y_pred"].to_numpy(dtype=int)
    wrong_values = (~sub["correct"].to_numpy(dtype=bool)).astype(int)

    positive_steps = np.diff(x)
    positive_steps = positive_steps[positive_steps > 0]
    nominal_step = float(np.nanmedian(positive_steps)) if len(positive_steps) else 1 / 60
    gap_threshold = max(1.0, nominal_step * 10)
    sample_width = min(max(nominal_step, 1 / 120), 0.25)

    def make_segments(values: np.ndarray) -> list[tuple[float, float, int]]:
        if len(values) == 0:
            return []
        segments = []
        start_idx = 0
        for idx in range(1, len(values)):
            is_gap = (x[idx] - x[idx - 1]) > gap_threshold
            if values[idx] != values[idx - 1] or is_gap:
                start = float(x[start_idx])
                end = float(x[idx - 1] + sample_width)
                if end > start:
                    segments.append((start, end, int(values[start_idx])))
                start_idx = idx
        start = float(x[start_idx])
        end = float(x[-1] + sample_width)
        if end > start:
            segments.append((start, end, int(values[start_idx])))
        return segments

    true_segments = make_segments(true_values)
    pred_segments = make_segments(pred_values)
    wrong_segments = [seg for seg in make_segments(wrong_values) if seg[2] == 1]

    window_min = 60.0
    max_elapsed = float(np.nanmax(x)) if len(x) else 0.0
    num_windows = max(1, int(np.ceil((max_elapsed + sample_width) / window_min)))
    fig_height = 1.75 * num_windows + 1.5
    fig, axes = plt.subplots(num_windows, 1, figsize=(16, fig_height), dpi=170, squeeze=False)
    axes_flat = axes[:, 0]

    def draw_segments(ax, segments: list[tuple[float, float, int]], y_base: float, height: float) -> None:
        left, right = ax.get_xlim()
        for start, end, class_id in segments:
            clipped_start = max(start, left)
            clipped_end = min(end, right)
            if clipped_end <= clipped_start:
                continue
            ax.broken_barh(
                [(clipped_start, clipped_end - clipped_start)],
                (y_base, height),
                facecolors=cmap(class_id),
                edgecolors="none",
                alpha=0.95,
            )

    for window_idx, ax in enumerate(axes_flat):
        start = window_idx * window_min
        end = min((window_idx + 1) * window_min, max(window_min, max_elapsed + sample_width))
        ax.set_xlim(start, end)
        ax.set_ylim(-0.2, 2.7)
        ax.axhspan(1.75, 2.35, color="#f7f7f7", zorder=0)
        ax.axhspan(0.85, 1.45, color="#f7f7f7", zorder=0)
        ax.axhspan(0.08, 0.43, color="#fff5f5", zorder=0)
        draw_segments(ax, true_segments, 1.82, 0.46)
        draw_segments(ax, pred_segments, 0.92, 0.46)
        for wrong_start, wrong_end, _ in wrong_segments:
            clipped_start = max(wrong_start, start)
            clipped_end = min(wrong_end, end)
            if clipped_end <= clipped_start:
                continue
            ax.broken_barh(
                [(clipped_start, clipped_end - clipped_start)],
                (0.13, 0.25),
                facecolors="#d62728",
                edgecolors="none",
                alpha=0.72,
            )
        in_window = (x >= start) & (x < end)
        wrong_rate = float(wrong_values[in_window].mean()) if in_window.any() else 0.0
        ax.text(
            0.995,
            0.1,
            f"window wrong={wrong_rate:.1%}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#8a1f1f",
        )
        ax.set_yticks([2.05, 1.15, 0.25])
        ax.set_yticklabels(["True", "Pred", "Wrong"])
        ax.grid(axis="x", alpha=0.18)
        ax.set_ylabel(f"{start:.0f}-{end:.0f} min", fontsize=9)
        if window_idx != num_windows - 1:
            ax.tick_params(axis="x", labelbottom=False)

    axes_flat[0].set_title(
        f"Trajectory {part['part']} temporal class prediction (accuracy={sub['correct'].mean():.1%})"
    )
    axes_flat[-1].set_xlabel("Elapsed time (min)")
    handles = [
        Line2D([0], [0], marker="s", color="w", label=f"{i}: {CLASS_SHORT[i]}",
               markerfacecolor=cmap(i), markersize=7)
        for i in range(11)
    ]
    handles.append(Line2D([0], [0], color="#d62728", linewidth=6, label="wrong sample band"))
    fig.legend(handles=handles, fontsize=8, loc="lower center", ncol=6, framealpha=0.95)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_dir / f"part{part['part']:02d}_true_vs_pred_timeline.png")
    plt.close(fig)


def plot_confusion(sub: pd.DataFrame, part: dict, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    apply_paper_style()
    counts = pd.crosstab(sub["y_true"], sub["y_pred"]).reindex(index=range(11), columns=range(11), fill_value=0)
    row_sums = counts.sum(axis=1).replace(0, np.nan)
    norm = counts.div(row_sums, axis=0).fillna(0.0)
    supports = counts.sum(axis=1).astype(int)
    x_labels = [f"{i}\n{CLASS_SHORT[i]}" for i in range(11)]
    y_labels = [f"{i} {CLASS_SHORT[i]}\n(n={supports.iloc[i]})" for i in range(11)]
    annot = counts.astype(str).to_numpy()

    fig, ax = plt.subplots(figsize=(8.6, 7.6), dpi=180)
    sns.heatmap(
        norm,
        ax=ax,
        cmap="Blues",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.35,
        linecolor="#f2f2f2",
        annot=annot,
        fmt="",
        annot_kws={"fontsize": 6.2},
        cbar_kws={"label": "Recall share within true class"},
    )
    ax.set_xticklabels(x_labels, rotation=0, ha="center")
    ax.set_yticklabels(y_labels, rotation=0)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Trajectory {part['part']} confusion matrix (accuracy={sub['correct'].mean():.1%})")
    ax.tick_params(axis="both", length=0)
    ax.text(
        0.0,
        -0.13,
        "Cell color is row-normalized recall share; cell text is sample count.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"part{part['part']:02d}_confusion_matrix.png")
    plt.close(fig)


def plot_class_distribution_comparison(sub: pd.DataFrame, part: dict, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    apply_paper_style()
    true_counts = sub["y_true"].value_counts().reindex(range(11), fill_value=0).sort_index()
    pred_counts = sub["y_pred"].value_counts().reindex(range(11), fill_value=0).sort_index()
    plot_df = pd.DataFrame({
        "Class": [f"{i} {CLASS_SHORT[i]}" for i in range(11)] * 2,
        "Samples": np.concatenate([true_counts.to_numpy(), pred_counts.to_numpy()]),
        "Distribution": ["True"] * 11 + ["Predicted"] * 11,
    })

    fig, ax = plt.subplots(figsize=(10.6, 5.4), dpi=180)
    sns.barplot(
        data=plot_df,
        x="Class",
        y="Samples",
        hue="Distribution",
        palette={"True": PAPER_PALETTE["true"], "Predicted": PAPER_PALETTE["predicted"]},
        ax=ax,
        edgecolor="#ffffff",
        linewidth=0.4,
    )
    ax.set_xlabel("Class")
    ax.set_ylabel("Samples")
    ax.set_title(f"Trajectory {part['part']} class distribution")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"part{part['part']:02d}_true_pred_class_distribution.png")
    plt.close(fig)


def plot_error_pair_lollipop(sub: pd.DataFrame, part: dict, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    apply_paper_style()
    wrong = sub[~sub["correct"]].copy()
    fig, ax = plt.subplots(figsize=(8.8, 5.8), dpi=180)
    if wrong.empty:
        ax.text(0.5, 0.5, "No errors", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_dir / f"part{part['part']:02d}_error_pair_lollipop.png")
        plt.close(fig)
        return

    pairs = (
        wrong.groupby(["y_true", "y_pred"])
        .size()
        .reset_index(name="Misclassified samples")
        .sort_values("Misclassified samples", ascending=False)
        .head(12)
    )
    pairs["Error pair"] = pairs.apply(
        lambda row: (
            f"{int(row['y_true'])} {CLASS_SHORT[int(row['y_true'])]}"
            f" -> {int(row['y_pred'])} {CLASS_SHORT[int(row['y_pred'])]}"
        ),
        axis=1,
    )
    pairs = pairs.sort_values("Misclassified samples", ascending=True).reset_index(drop=True)
    palette = sns.color_palette("Reds", n_colors=len(pairs) + 3)[3:]
    y_pos = np.arange(len(pairs))
    values = pairs["Misclassified samples"].to_numpy()

    ax.hlines(y=y_pos, xmin=0, xmax=values, color="#c9c9c9", linewidth=2.0, zorder=1)
    ax.scatter(values, y_pos, s=120, color=palette, edgecolor="#5f1f1f", linewidth=0.7, zorder=2)
    for idx, value in enumerate(values):
        ax.text(value, idx, f" {int(value)}", va="center", ha="left", fontsize=8.5, color="#333333")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pairs["Error pair"])
    ax.set_xlabel("Misclassified samples")
    ax.set_ylabel("True class -> predicted class")
    ax.set_title(f"Trajectory {part['part']} dominant misclassification pairs")
    ax.set_xlim(0, max(values) * 1.12)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"part{part['part']:02d}_error_pair_lollipop.png")
    plt.close(fig)


def plot_error_pair_heatmap(sub: pd.DataFrame, part: dict, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    apply_paper_style()
    wrong = sub[~sub["correct"]].copy()
    if wrong.empty:
        return
    counts = pd.crosstab(wrong["y_true"], wrong["y_pred"]).reindex(index=range(11), columns=range(11), fill_value=0)
    for class_id in range(11):
        counts.iloc[class_id, class_id] = 0
    nonzero_rows = counts.sum(axis=1) > 0
    nonzero_cols = counts.sum(axis=0) > 0
    counts = counts.loc[nonzero_rows, nonzero_cols]
    if counts.empty:
        return
    row_labels = [f"{int(i)} {CLASS_SHORT[int(i)]}" for i in counts.index]
    col_labels = [f"{int(i)} {CLASS_SHORT[int(i)]}" for i in counts.columns]

    fig, ax = plt.subplots(figsize=(8.0, 5.8), dpi=180)
    sns.heatmap(
        counts,
        ax=ax,
        cmap="Reds",
        annot=True,
        fmt="d",
        linewidths=0.35,
        linecolor="#f2f2f2",
        cbar_kws={"label": "Misclassified samples"},
    )
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticklabels(row_labels, rotation=0)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Trajectory {part['part']} misclassification heatmap")
    fig.tight_layout()
    fig.savefig(output_dir / f"part{part['part']:02d}_error_pair_heatmap.png")
    plt.close(fig)


def plot_date_overview(summary_rows: list[dict], date: str, output_dir: Path) -> None:
    if not summary_rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    apply_paper_style()
    summary = pd.DataFrame(summary_rows).sort_values("part")
    summary["Trajectory"] = summary.apply(
        lambda row: f"Trajectory {int(row['part'])}\n(n={int(row['rows'])}, errors={int(row['wrong'])})",
        axis=1,
    )
    summary["Error rate"] = summary["wrong"].astype(float) / summary["rows"].clip(lower=1).astype(float)
    plot_df = summary.melt(
        id_vars=["Trajectory", "rows", "wrong"],
        value_vars=["acc", "Error rate"],
        var_name="Metric",
        value_name="Rate",
    )
    plot_df["Metric"] = plot_df["Metric"].replace({"acc": "Accuracy"})

    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180)
    sns.barplot(
        data=plot_df,
        y="Trajectory",
        x="Rate",
        hue="Metric",
        palette={"Accuracy": PAPER_PALETTE["accuracy"], "Error rate": PAPER_PALETTE["error_rate"]},
        ax=ax,
        edgecolor="#ffffff",
        linewidth=0.5,
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Rate")
    ax.set_ylabel("Trajectory")
    ax.set_title(f"Trajectory-level classification performance ({date})")
    ax.legend(title="")
    ax.grid(axis="y", visible=False)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=2, fontsize=8, labels=[f"{v.get_width() * 100:.1f}%" for v in container])
    fig.tight_layout()
    fig.savefig(output_dir / f"{date}_part_accuracy_overview.png")
    plt.close(fig)


def plot_part(df: pd.DataFrame, part: dict, output_dir: Path) -> dict | None:
    sub = subset_for_part(df, part)
    if sub.empty:
        return None
    plot_local_error_map(sub, part, output_dir)
    plot_spatial_errors_by_part(sub, part, output_dir)
    plot_timeline(sub, part, output_dir)
    plot_confusion(sub, part, output_dir)
    plot_class_distribution_comparison(sub, part, output_dir)
    plot_error_pair_lollipop(sub, part, output_dir)
    plot_error_pair_heatmap(sub, part, output_dir)
    wrong = sub[~sub["correct"]]
    top_errors = []
    if not wrong.empty:
        for (y_true, y_pred), count in wrong.groupby(["y_true", "y_pred"]).size().sort_values(ascending=False).head(8).items():
            top_errors.append(f"{y_true}->{y_pred}:{int(count)}")
    return {
        "part": part["part"],
        "name": part["name"],
        "start": str(part["start"]),
        "end": str(part["end"]),
        "rows": int(len(sub)),
        "acc": float(sub["correct"].mean()),
        "macro_recall_observed_classes": float(sub.groupby("y_true")["correct"].mean().mean()),
        "wrong": int((~sub["correct"]).sum()),
        "top_errors": "; ".join(top_errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--part-dir", default="/private/data/B_deep_part")
    parser.add_argument("--date", default="2024-10-27")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    predictions = resolve(args.predictions)
    output_dir = resolve(args.output_dir) if args.output_dir else predictions.parent / "part_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(predictions, encoding="utf-8-sig")
    df["frame_time"] = pd.to_datetime(df["frame_time"], errors="coerce")
    for col in ["经度", "纬度"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["frame_time", "经度", "纬度"])

    ranges = load_part_ranges(resolve(args.part_dir), args.date)
    summary_rows = []
    for part in ranges:
        row = plot_part(df, part, output_dir)
        if row is not None:
            summary_rows.append(row)
    plot_date_overview(summary_rows, args.date, output_dir)
    pd.DataFrame(summary_rows).to_csv(output_dir / f"{args.date}_part_diagnostic_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Saved part diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
