#!/usr/bin/env python3
"""Build paper figures for the trimodal fusion ablation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-agri-mbt")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = PROJECT_ROOT / "assets"

CLASS_SHORT = ["R-EH", "S-EH", "T-EH", "Full", "R-Tr", "S-Tr", "T-Tr", "Off", "Idle", "Unload", "Road"]
MODEL_DIRS = {
    "AST": PROJECT_ROOT / "artifacts" / "paper_results" / "ast",
    "ViT": PROJECT_ROOT / "artifacts" / "paper_results" / "image_best",
    "BiLSTM": PROJECT_ROOT / "artifacts" / "paper_results" / "trnet",
    "TIM concat": PROJECT_ROOT / "artifacts" / "paper_results" / "trimodal_concat",
    "TIM class-gate": PROJECT_ROOT / "artifacts" / "paper_results" / "trimodal_class_gate",
}
MODEL_ORDER = ["AST", "ViT", "BiLSTM", "TIM concat", "TIM class-gate"]
MODEL_COLORS = {
    "AST": "#4c78a8",
    "ViT": "#f58518",
    "BiLSTM": "#54a24b",
    "TIM concat": "#8e6c8a",
    "TIM class-gate": "#c84e4e",
}


def apply_style() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
    plt.rcParams.update({
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "axes.titlecolor": "#222222",
        "axes.titleweight": "semibold",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "grid.color": "#dddddd",
        "grid.linewidth": 0.55,
        "legend.frameon": True,
        "legend.framealpha": 0.94,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
    })


def load_summary() -> pd.DataFrame:
    rows = []
    for model, path in MODEL_DIRS.items():
        summary = json.loads((path / "summary.json").read_text())
        test = summary["test"]
        rows.append({
            "model": model,
            "Accuracy": test["acc"] * 100.0,
            "Macro-F1": test["macro_f1"] * 100.0,
            "Weighted-F1": test["weighted_f1"] * 100.0,
            "Val macro-F1": summary["best_val_macro_f1"] * 100.0,
            "Test loss": test["loss"],
        })
    return pd.DataFrame(rows)


def load_per_class() -> pd.DataFrame:
    frames = []
    for model, path in MODEL_DIRS.items():
        df = pd.read_csv(path / "per_class_metrics.csv")
        df["model"] = model
        df["class"] = df["class_id"].map(lambda x: CLASS_SHORT[int(x)])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_confusion(model: str) -> np.ndarray:
    summary = json.loads((MODEL_DIRS[model] / "summary.json").read_text())
    return np.asarray(summary["test"]["confusion_matrix"], dtype=int)


def save_overall_results(summary: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    long = summary.melt(
        id_vars="model",
        value_vars=["Accuracy", "Macro-F1", "Weighted-F1"],
        var_name="Metric",
        value_name="Score",
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    sns.barplot(
        data=long,
        x="Metric",
        y="Score",
        hue="model",
        hue_order=MODEL_ORDER,
        palette=[MODEL_COLORS[m] for m in MODEL_ORDER],
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 88)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16), title="")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_modality_roles(per_class: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    f1 = per_class.pivot(index="class_id", columns="model", values="f1").loc[:, MODEL_ORDER] * 100.0
    f1.index = [CLASS_SHORT[int(idx)] for idx in f1.index]
    best_single = f1[["AST", "ViT", "BiLSTM"]].max(axis=1)
    gate_gain = f1["TIM class-gate"] - best_single

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.0, 7.8),
        gridspec_kw={"height_ratios": [1.85, 1.65], "hspace": 0.40},
    )
    sns.heatmap(
        f1.T,
        ax=axes[0],
        cmap="YlGnBu",
        annot=True,
        fmt=".1f",
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "F1 (%)"},
        vmin=0,
        vmax=100,
    )
    axes[0].set_xlabel("Operation class")
    axes[0].set_ylabel("Model")
    axes[0].set_title("Per-class F1 across modalities and fusion models")
    axes[0].grid(False)

    colors = ["#3b7f5f" if value >= 0 else "#b85c5c" for value in gate_gain]
    bars = axes[1].bar(gate_gain.index, gate_gain.values, color=colors, edgecolor="#333333", linewidth=0.45)
    axes[1].axhline(0, color="#333333", linewidth=0.9, zorder=0)
    axes[1].set_ylabel("F1 gain (points)")
    axes[1].set_xlabel("Operation class")
    axes[1].set_title("TIM class-gate minus best single-modality F1")
    axes[1].grid(False)
    axes[1].set_ylim(min(-6.2, gate_gain.min() - 2.3), max(16.0, gate_gain.max() + 2.6))
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    for tick in axes[1].get_xticklabels():
        tick.set_rotation(0)
    for bar, value in zip(bars, gate_gain.values):
        xpos = bar.get_x() + bar.get_width() / 2.0
        if value >= 0:
            ypos = value + 0.85
            va = "bottom"
        else:
            ypos = value - 0.85
            va = "top"
        axes[1].text(xpos, ypos, f"{value:+.1f}", ha="center", va=va, fontsize=8.5, clip_on=False)
    fig.subplots_adjust(left=0.105, right=0.92, top=0.93, bottom=0.08, hspace=0.42)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_fusion_ablation(summary: pd.DataFrame, per_class: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))
    axes = axes.ravel()

    # Aggregate metric ablation.
    metrics = ["Accuracy", "Macro-F1", "Weighted-F1"]
    agg = summary[summary["model"].isin(["TIM concat", "TIM class-gate"])].melt(
        id_vars="model",
        value_vars=metrics,
        var_name="Metric",
        value_name="Score",
    )
    agg["Fusion"] = agg["model"].map({"TIM concat": "Concat", "TIM class-gate": "Class-gate"})
    sns.barplot(
        data=agg,
        x="Metric",
        y="Score",
        hue="Fusion",
        hue_order=["Concat", "Class-gate"],
        palette=[MODEL_COLORS["TIM concat"], MODEL_COLORS["TIM class-gate"]],
        ax=axes[0],
    )
    axes[0].set_title("Aggregate fusion ablation")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Score (%)")
    axes[0].set_ylim(55, 84)
    axes[0].legend(title="", loc="lower right")
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.1f", padding=2, fontsize=8)

    # Per-class delta.
    f1 = per_class.pivot(index="class_id", columns="model", values="f1").loc[:, ["TIM concat", "TIM class-gate"]] * 100.0
    f1.index = [CLASS_SHORT[int(idx)] for idx in f1.index]
    delta = f1["TIM class-gate"] - f1["TIM concat"]
    colors = ["#3b7f5f" if value >= 0 else "#b85c5c" for value in delta]
    axes[1].bar(delta.index, delta.values, color=colors, edgecolor="#333333", linewidth=0.45)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_title("Per-class F1 change")
    axes[1].set_ylabel("Class-gate - concat (points)")
    axes[1].set_xlabel("Operation class")
    for tick in axes[1].get_xticklabels():
        tick.set_rotation(35)
        tick.set_ha("right")

    # Largest error-pair reductions.
    concat = load_confusion("TIM concat")
    gate = load_confusion("TIM class-gate")
    changes = []
    for i in range(len(CLASS_SHORT)):
        for j in range(len(CLASS_SHORT)):
            if i == j:
                continue
            reduction = int(concat[i, j] - gate[i, j])
            if reduction > 0:
                changes.append({
                    "pair": f"{CLASS_SHORT[i]} -> {CLASS_SHORT[j]}",
                    "reduction": reduction,
                })
    reductions = pd.DataFrame(changes).sort_values("reduction", ascending=False).head(8)
    sns.barplot(data=reductions, x="reduction", y="pair", color="#3b7f5f", ax=axes[2])
    axes[2].set_title("Largest wrong-pair reductions")
    axes[2].set_xlabel("Fewer wrong samples")
    axes[2].set_ylabel("True -> predicted")
    for container in axes[2].containers:
        axes[2].bar_label(container, fmt="%.0f", padding=3, fontsize=8)

    # Validation/test comparison for the two trimodal variants.
    fusion_rows = summary[summary["model"].isin(["TIM concat", "TIM class-gate"])].copy()
    fusion_rows["Fusion"] = fusion_rows["model"].map({"TIM concat": "Concat", "TIM class-gate": "Class-gate"})
    comparison = fusion_rows.melt(
        id_vars="Fusion",
        value_vars=["Val macro-F1", "Macro-F1", "Weighted-F1"],
        var_name="Metric",
        value_name="Score",
    )
    sns.barplot(
        data=comparison,
        x="Metric",
        y="Score",
        hue="Fusion",
        hue_order=["Concat", "Class-gate"],
        palette=[MODEL_COLORS["TIM concat"], MODEL_COLORS["TIM class-gate"]],
        ax=axes[3],
    )
    axes[3].set_title("Trimodal validation and test metrics")
    axes[3].set_xlabel("")
    axes[3].set_ylabel("Score (%)")
    axes[3].legend(title="", loc="lower right")
    for container in axes[3].containers:
        axes[3].bar_label(container, fmt="%.1f", padding=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(FIGURE_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    summary = load_summary()
    per_class = load_per_class()
    save_overall_results(summary, output_dir / "fig_overall_results.png")
    save_modality_roles(per_class, output_dir / "fig_modality_roles.png")
    save_fusion_ablation(summary, per_class, output_dir / "fig_fusion_ablation.png")
    print(f"Saved fusion figures to {output_dir}")


if __name__ == "__main__":
    main()
