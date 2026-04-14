#!/usr/bin/env python3
"""
Analyze trajectory/image/multimodal ablation results.
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
MODE_LABELS = {
    "trajectory_only": "Trajectory Only",
    "image_only": "Image Only",
    "multimodal": "Multimodal",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze ablation suite")
    parser.add_argument("--suite-dir", default="experiments/ablation_20241018_suite")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_summary(suite_dir: Path, mode: str) -> dict:
    path = suite_dir / mode / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def best_epoch(summary: dict) -> dict:
    return max(summary["history"], key=lambda item: item["val"]["macro_f1"])


def final_epoch(summary: dict) -> dict:
    return summary["history"][-1]


def class_metrics(summary: dict, mode: str) -> pd.DataFrame:
    conf = np.asarray(summary["test"]["confusion_matrix"], dtype=np.float64)
    rows = []
    class_names = summary.get("class_names", [str(i) for i in range(conf.shape[0])])
    for i in range(conf.shape[0]):
        tp = conf[i, i]
        support = conf[i, :].sum()
        predicted = conf[:, i].sum()
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "mode": mode,
                "class_id": i,
                "class_name": class_names[i] if i < len(class_names) else str(i),
                "support": int(support),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def plot_overall(comparison: pd.DataFrame, output_dir: Path):
    metrics = ["best_val_macro_f1", "test_macro_f1", "test_weighted_f1", "test_acc"]
    labels = ["Best Val Macro F1", "Test Macro F1", "Test Weighted F1", "Test Acc"]
    x = np.arange(len(comparison))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 5), dpi=160)
    for offset, metric, label in zip(np.linspace(-1.5, 1.5, len(metrics)) * width, metrics, labels):
        ax.bar(x + offset, comparison[metric], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in comparison["mode"]])
    ax.set_ylim(0, 1)
    ax.set_title("Ablation Overall Metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "overall_metrics.png")
    plt.close(fig)


def plot_curves(summaries: dict, output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=160)
    specs = [("loss", "Loss"), ("acc", "Accuracy"), ("macro_f1", "Macro F1")]
    for ax, (metric, title) in zip(axes, specs):
        for mode, summary in summaries.items():
            epochs = [item["epoch"] for item in summary["history"]]
            ax.plot(epochs, [item["val"][metric] for item in summary["history"]], label=f"{MODE_LABELS[mode]} val")
            if metric == "macro_f1":
                ax.plot(epochs, [item["train"][metric] for item in summary["history"]], linestyle="--", alpha=0.55, label=f"{MODE_LABELS[mode]} train")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves_comparison.png")
    plt.close(fig)


def plot_class_metrics(class_df: pd.DataFrame, output_dir: Path):
    for metric in ["recall", "f1", "precision"]:
        pivot = class_df.pivot(index="class_id", columns="mode", values=metric).reindex(columns=MODES)
        fig, ax = plt.subplots(figsize=(12, 4.5), dpi=160)
        x = np.arange(len(pivot))
        width = 0.25
        for idx, mode in enumerate(MODES):
            ax.bar(x + (idx - 1) * width, pivot[mode], width, label=MODE_LABELS[mode])
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Class ID")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"Test {metric.capitalize()} by Class")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"per_class_{metric}_comparison.png")
        plt.close(fig)


def plot_confusions(summaries: dict, output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=160)
    for ax, mode in zip(axes, MODES):
        mat = np.asarray(summaries[mode]["test"]["confusion_matrix"], dtype=np.float64)
        norm = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1.0)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(MODE_LABELS[mode])
        ax.set_xlabel("Pred")
        ax.set_ylabel("True")
        ax.set_xticks(range(mat.shape[0]))
        ax.set_yticks(range(mat.shape[0]))
        ax.set_xticklabels(range(mat.shape[0]))
        ax.set_yticklabels(range(mat.shape[0]))
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.savefig(output_dir / "confusion_matrices_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_spatial_error_density(suite_dir: Path, summaries: dict, output_dir: Path):
    pred_paths = {mode: suite_dir / mode / "predictions.csv" for mode in MODES}
    if not all(path.exists() for path in pred_paths.values()):
        print("Skip spatial error density: missing predictions.csv files")
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
        acc = summaries[mode]["test"]["acc"] * 100
        ax.set_title(f"{MODE_LABELS[mode]}\nError Distribution (Acc: {acc:.2f}%)", fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.18)
    if last_hex is not None:
        fig.colorbar(last_hex, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="Error Count")
    fig.suptitle("Spatial Distribution of Prediction Errors\n(Red = High Error Density)", fontweight="bold", y=1.02)
    fig.savefig(output_dir / "spatial_error_density.png", bbox_inches="tight")
    plt.close(fig)


def write_report(comparison: pd.DataFrame, class_df: pd.DataFrame, summaries: dict, output_dir: Path):
    best = comparison.sort_values("test_macro_f1", ascending=False).iloc[0]
    display_cols = [
        "mode",
        "best_epoch",
        "best_val_macro_f1",
        "final_train_macro_f1",
        "final_val_macro_f1",
        "test_macro_f1",
        "test_weighted_f1",
        "test_acc",
    ]
    table_df = comparison[display_cols].copy()
    for col in table_df.columns:
        if pd.api.types.is_float_dtype(table_df[col]):
            table_df[col] = table_df[col].map(lambda x: f"{x:.4f}")
    markdown_table = table_df.to_csv(sep="|", index=False, lineterminator="\n")
    markdown_lines = markdown_table.strip().split("\n")
    markdown_table = "\n".join([
        f"|{markdown_lines[0]}|",
        "|" + "|".join(["---"] * len(table_df.columns)) + "|",
        *[f"|{line}|" for line in markdown_lines[1:]],
    ])
    lines = [
        "# Ablation Analysis",
        "",
        "## Overall Results",
        "",
        markdown_table,
        "",
        f"Best test macro F1: **{MODE_LABELS[best['mode']]}** ({best['test_macro_f1']:.4f}).",
        "",
        "## Generalization",
        "",
    ]
    for _, row in comparison.iterrows():
        lines.append(
            f"- {MODE_LABELS[row['mode']]}: best epoch {int(row['best_epoch'])}, "
            f"final train-val macro F1 gap {row['final_train_val_macro_f1_gap']:.4f}."
        )

    lines.extend([
        "",
        "## Per-Class Weak Spots",
        "",
    ])
    for mode in MODES:
        sub = class_df[class_df["mode"] == mode].sort_values("f1").head(5)
        weak = ", ".join(
            f"class {int(r.class_id)} F1={r.f1:.3f} support={int(r.support)}"
            for r in sub.itertuples()
        )
        lines.append(f"- {MODE_LABELS[mode]} weakest classes: {weak}")

    lines.extend([
        "",
        "## Sampling Note",
        "",
        "By default the new suite uses causal trajectory context and dense validation/test stride. "
        "Adaptive sampling changes the density of training anchors by behavior duration; it does not require the true label at prediction time. "
        "Images are sampled around the prediction anchor, e.g. `--image-sampling center --image-radius 8 --image-window-size 9`. "
        "If short visual events are still missed, reduce `--eval-stride`, increase `--image-window-size`, or add a short-window branch.",
        "",
        "## Artifacts",
        "",
        "- `comparison.csv`",
        "- `per_class_metrics.csv`",
        "- `overall_metrics.png`",
        "- `training_curves_comparison.png`",
        "- `per_class_recall_comparison.png`",
        "- `per_class_f1_comparison.png`",
        "- `per_class_precision_comparison.png`",
        "- `confusion_matrices_comparison.png`",
    ])
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    suite_dir = resolve(args.suite_dir)
    output_dir = resolve(args.output_dir) if args.output_dir else suite_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {mode: load_summary(suite_dir, mode) for mode in MODES}
    comparison_rows = []
    class_frames = []
    for mode, summary in summaries.items():
        best = best_epoch(summary)
        final = final_epoch(summary)
        test = summary["test"]
        comparison_rows.append(
            {
                "mode": mode,
                "best_epoch": int(best["epoch"]),
                "best_val_macro_f1": float(summary["best_val_macro_f1"]),
                "final_train_macro_f1": float(final["train"]["macro_f1"]),
                "final_val_macro_f1": float(final["val"]["macro_f1"]),
                "final_train_val_macro_f1_gap": float(final["train"]["macro_f1"] - final["val"]["macro_f1"]),
                "test_macro_f1": float(test["macro_f1"]),
                "test_weighted_f1": float(test["weighted_f1"]),
                "test_acc": float(test["acc"]),
                "test_loss": float(test["loss"]),
            }
        )
        class_frames.append(class_metrics(summary, mode))

    comparison = pd.DataFrame(comparison_rows)
    class_df = pd.concat(class_frames, ignore_index=True)
    comparison.to_csv(output_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    class_df.to_csv(output_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")

    plot_overall(comparison, output_dir)
    plot_curves(summaries, output_dir)
    plot_class_metrics(class_df, output_dir)
    plot_confusions(summaries, output_dir)
    plot_spatial_error_density(suite_dir, summaries, output_dir)
    write_report(comparison, class_df, summaries, output_dir)

    print(f"Analysis saved to: {output_dir}")


if __name__ == "__main__":
    main()
