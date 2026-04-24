#!/usr/bin/env python3
"""Build method-section schematic figures for the manuscript."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-agri-mbt")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def add_box(ax, xy, width, height, text, face, edge="#2c2c2c", lw=1.1, fontsize=10, weight="normal"):
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#202020",
        linespacing=1.18,
    )
    return box


def add_arrow(ax, start, end, color="#444444", lw=1.2):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=2, shrinkB=2, mutation_scale=10),
    )


def draw_pipeline(ax):
    colors = {
        "raw": "#f4f7fb",
        "sync": "#eef6f1",
        "window": "#fff7e8",
        "traj": "#e8f2ff",
        "image": "#edf7ea",
        "audio": "#fff0f0",
        "feature": "#f7f4ff",
        "head": "#f1f1f1",
    }

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(0.2, 6.65, "Data Processing and Temporal Alignment", fontsize=17, fontweight="semibold", color="#202020")
    ax.text(
        0.2,
        6.26,
        "All modalities are aligned at the same label anchor; only causal evidence before the anchor is used.",
        fontsize=10.5,
        color="#555555",
    )

    add_box(ax, (0.25, 5.05), 1.55, 0.68, "Raw GNSS\nrecords", colors["raw"], fontsize=9.5, weight="semibold")
    add_box(ax, (0.25, 4.15), 1.55, 0.68, "ViT\nframes", colors["raw"], fontsize=9.5, weight="semibold")
    add_box(ax, (0.25, 3.25), 1.55, 0.68, "Audio\nclips", colors["raw"], fontsize=9.5, weight="semibold")
    add_box(ax, (2.25, 4.06), 1.75, 1.06, "Timestamp\ncleaning and\nsynchronization", colors["sync"], fontsize=9.5, weight="semibold")
    add_box(ax, (4.5, 4.06), 1.75, 1.06, "Label anchor\nat time $\\tau$", colors["window"], fontsize=10, weight="semibold")
    add_box(ax, (6.8, 5.05), 1.72, 0.68, "128-s causal\ntrajectory window", colors["traj"], fontsize=9.2, weight="semibold")
    add_box(ax, (6.8, 4.15), 1.72, 0.68, "9 nearest\ncausal frames", colors["image"], fontsize=9.2, weight="semibold")
    add_box(ax, (6.8, 3.25), 1.72, 0.68, "1-s aligned\naudio segment", colors["audio"], fontsize=9.2, weight="semibold")
    add_box(ax, (9.0, 5.05), 2.05, 0.68, "36-D kinematic\nfeature sequence", colors["feature"], fontsize=9.2, weight="semibold")
    add_box(ax, (9.0, 4.15), 2.05, 0.68, "Frame embeddings\nand temporal deltas", colors["feature"], fontsize=9.2, weight="semibold")
    add_box(ax, (9.0, 3.25), 2.05, 0.68, "Log-mel\nspectrogram", colors["feature"], fontsize=9.2, weight="semibold")
    add_box(ax, (9.0, 2.05), 2.05, 0.68, "Matched sample\nfor one label", colors["head"], fontsize=9.2, weight="semibold")

    for y in [5.39, 4.49, 3.59]:
        add_arrow(ax, (1.82, y), (2.23, 4.59))
    add_arrow(ax, (4.02, 4.59), (4.48, 4.59))
    for y in [5.39, 4.49, 3.59]:
        add_arrow(ax, (6.27, 4.59), (6.78, y))
        add_arrow(ax, (8.54, y), (8.98, y))
    for y in [5.39, 4.49, 3.59]:
        add_arrow(ax, (10.04, y - 0.37), (10.04, 2.75))

    # Causal time axis.
    y = 1.15
    ax.plot([1.0, 10.9], [y, y], color="#333333", lw=1.3)
    ax.annotate("", xy=(10.9, y), xytext=(10.55, y), arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#333333"))
    ax.text(0.75, y - 0.35, "$\\tau - 128$s", fontsize=9.5, ha="center", color="#333333")
    ax.text(10.9, y - 0.35, "$\\tau$", fontsize=9.5, ha="center", color="#333333")
    ax.axvline(10.45, ymin=0.07, ymax=0.29, color="#ba2f2f", lw=1.2, ls="--")
    ax.text(10.45, 1.53, "anchor", fontsize=9, ha="center", color="#8a1f1f")

    traj_x = np.linspace(1.0, 10.45, 24)
    ax.scatter(traj_x, np.full_like(traj_x, y + 0.28), s=18, color="#4c78a8", alpha=0.85, zorder=3)
    frame_x = np.linspace(9.25, 10.45, 9)
    for x in frame_x:
        ax.add_patch(patches.Rectangle((x - 0.035, y + 0.55), 0.07, 0.28, facecolor="#59a14f", edgecolor="white", lw=0.35))
    ax.add_patch(patches.Rectangle((10.32, y - 0.08), 0.25, 0.16, facecolor="#e15759", edgecolor="none", alpha=0.85))
    ax.text(5.7, y + 0.55, "causal window", fontsize=9.3, color="#555555", ha="center")
    ax.text(9.86, y + 0.95, "selected frames", fontsize=8.7, color="#4d6f3c", ha="center")
    ax.text(10.45, y - 0.30, "audio clip", fontsize=8.7, color="#9f3030", ha="center")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="assets/fig_method_data_processing.png")
    args = parser.parse_args()

    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.6, 8.0), dpi=220)
    draw_pipeline(ax)
    fig.tight_layout(pad=0.6)
    fig.savefig(output)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
