#!/usr/bin/env python3
"""
Analyze old MBT train_test.py outputs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-agri-mbt")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODES = ["trajectory_only", "image_only", "multimodal"]
LABELS = {
    "trajectory_only": "Trajectory Only",
    "image_only": "Image Only",
    "multimodal": "Multimodal",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze old MBT suite")
    parser.add_argument("--results-dir", default="experiments/old_mbt_20241018_suite")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def find_result(results_dir: Path, mode: str) -> Path:
    matches = sorted(results_dir.glob(f"results_{mode}_*.json"))
    if not matches:
        matches = sorted(results_dir.glob(f"results_{mode}.json"))
    if not matches:
        raise FileNotFoundError(f"No result JSON for mode {mode} in {results_dir}")
    return matches[-1]


def load_results(results_dir: Path):
    out = {}
    for mode in MODES:
        path = find_result(results_dir, mode)
        with open(path, "r", encoding="utf-8") as f:
            out[mode] = json.load(f)
        out[mode]["_path"] = str(path)
    return out


def comparison_df(results: dict) -> pd.DataFrame:
    rows = []
    for mode, item in results.items():
        metrics = item["metrics"]
        rows.append(
            {
                "mode": mode,
                "best_val_acc": item["best_val_acc"],
                "final_train_acc": item["final_train_acc"],
                "final_val_acc": item["final_val_acc"],
                "macro_precision": metrics["macro_avg"]["precision"],
                "macro_recall": metrics["macro_avg"]["recall"],
                "macro_f1": metrics["macro_avg"]["f1_score"],
                "weighted_f1": metrics["weighted_avg"]["f1_score"],
                "result_json": item["_path"],
            }
        )
    return pd.DataFrame(rows)


def per_class_df(results: dict) -> pd.DataFrame:
    rows = []
    for mode, item in results.items():
        for class_key, vals in item["metrics"]["per_class"].items():
            class_id = int(class_key.split("_")[1])
            rows.append(
                {
                    "mode": mode,
                    "class_id": class_id,
                    "precision": vals["precision"],
                    "recall": vals["recall"],
                    "f1_score": vals["f1_score"],
                }
            )
    return pd.DataFrame(rows)


def plot_overall(df: pd.DataFrame, output_dir: Path):
    metrics = ["best_val_acc", "macro_f1", "weighted_f1"]
    x = range(len(df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    for i, metric in enumerate(metrics):
        ax.bar([v + (i - 1) * width for v in x], df[metric], width, label=metric)
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABELS[m] for m in df["mode"]])
    ax.set_ylim(0, 100)
    ax.set_title("Old MBT Overall Metrics")
    ax.set_ylabel("%")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "overall_metrics.png")
    plt.close(fig)


def plot_curves(results: dict, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160)
    for mode, item in results.items():
        hist = item["history"]
        epochs = range(1, len(hist["train_acc"]) + 1)
        axes[0].plot(epochs, hist["train_acc"], linestyle="--", alpha=0.6, label=f"{LABELS[mode]} train")
        axes[0].plot(epochs, hist["val_acc"], label=f"{LABELS[mode]} val")
        axes[1].plot(epochs, hist["train_loss"], linestyle="--", alpha=0.6, label=f"{LABELS[mode]} train")
        axes[1].plot(epochs, hist["val_loss"], label=f"{LABELS[mode]} val")
    axes[0].set_title("Accuracy")
    axes[1].set_title("Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves_comparison.png")
    plt.close(fig)


def plot_per_class(df: pd.DataFrame, output_dir: Path):
    for metric in ["recall", "f1_score", "precision"]:
        pivot = df.pivot(index="class_id", columns="mode", values=metric).reindex(columns=MODES)
        fig, ax = plt.subplots(figsize=(12, 4.5), dpi=160)
        x = range(len(pivot))
        width = 0.25
        for i, mode in enumerate(MODES):
            ax.bar([v + (i - 1) * width for v in x], pivot[mode], width, label=LABELS[mode])
        ax.set_xticks(list(x))
        ax.set_xticklabels(pivot.index)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Class ID")
        ax.set_ylabel("%")
        ax.set_title(f"Per-class {metric}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"per_class_{metric}.png")
        plt.close(fig)


def find_predictions(results_dir: Path, mode: str) -> Path | None:
    matches = sorted(results_dir.glob(f"predictions_{mode}_*.csv"))
    return matches[-1] if matches else None


def plot_spatial_error_density(results_dir: Path, results: dict, output_dir: Path):
    pred_paths = {mode: find_predictions(results_dir, mode) for mode in MODES}
    if not all(pred_paths.values()):
        print("Skip spatial error density: missing predictions CSV files")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=160, sharex=True, sharey=True)
    last_hex = None
    for ax, mode in zip(axes, MODES):
        df = pd.read_csv(pred_paths[mode], encoding="utf-8-sig")
        if not {"经度", "纬度", "correct"}.issubset(df.columns):
            continue
        correct = df["correct"].astype(bool)
        ax.scatter(df.loc[correct, "经度"], df.loc[correct, "纬度"], c="#7ee787", s=3, alpha=0.25, linewidths=0)
        wrong = df.loc[~correct]
        if len(wrong) > 0:
            last_hex = ax.hexbin(
                wrong["经度"],
                wrong["纬度"],
                gridsize=30,
                cmap="Reds",
                mincnt=1,
                alpha=0.72,
            )
        acc = float(results[mode].get("best_checkpoint_val_acc", results[mode].get("final_val_acc", 0.0)))
        ax.set_title(f"{LABELS[mode]}\nError Distribution (Acc: {acc:.2f}%)", fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.18)
    if last_hex is not None:
        fig.colorbar(last_hex, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="Error Count")
    fig.suptitle("Spatial Distribution of Prediction Errors\n(Red = High Error Density)", fontweight="bold", y=1.02)
    fig.savefig(output_dir / "spatial_error_density.png", bbox_inches="tight")
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    use = df[["mode", "best_val_acc", "macro_f1", "weighted_f1", "final_val_acc"]].copy()
    for col in use.columns:
        if pd.api.types.is_float_dtype(use[col]):
            use[col] = use[col].map(lambda x: f"{x:.2f}")
    csv_text = use.to_csv(sep="|", index=False, lineterminator="\n")
    lines = csv_text.strip().split("\n")
    return "\n".join([
        f"|{lines[0]}|",
        "|" + "|".join(["---"] * len(use.columns)) + "|",
        *[f"|{line}|" for line in lines[1:]],
    ])


def write_report(comp: pd.DataFrame, per_class: pd.DataFrame, output_dir: Path):
    best = comp.sort_values("macro_f1", ascending=False).iloc[0]
    lines = [
        "# Old MBT Suite Analysis",
        "",
        "## Overall",
        "",
        markdown_table(comp),
        "",
        f"Best macro F1: **{LABELS[best['mode']]}** ({best['macro_f1']:.2f}%).",
        "",
        "## Weak Classes",
        "",
    ]
    for mode in MODES:
        sub = per_class[per_class["mode"] == mode].sort_values("f1_score").head(5)
        weak = ", ".join(f"C{int(r.class_id)} F1={r.f1_score:.2f}" for r in sub.itertuples())
        lines.append(f"- {LABELS[mode]}: {weak}")
    lines.extend([
        "",
        "## Note",
        "",
        "This analysis follows the old MBT training code. It uses row-level random train/test split inside `train_test.py`, not the explicit temporal split used by the newer ablation script.",
    ])
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    results_dir = resolve(args.results_dir)
    output_dir = resolve(args.output_dir) if args.output_dir else results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir)
    comp = comparison_df(results)
    per_class = per_class_df(results)
    comp.to_csv(output_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(output_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")
    plot_overall(comp, output_dir)
    plot_curves(results, output_dir)
    plot_per_class(per_class, output_dir)
    plot_spatial_error_density(results_dir, results, output_dir)
    write_report(comp, per_class, output_dir)
    print(f"Analysis saved to: {output_dir}")


if __name__ == "__main__":
    main()
