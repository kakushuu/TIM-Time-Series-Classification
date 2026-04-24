import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns


ROOT = Path("/private/research/Agri-MBT")
DATA_DIR = ROOT / "data/b_deep_part_multimodal_full_clean_20260417"
RESULT_DIR = ROOT / "logs/paper_4090_final_seed44/results"
FIG_DIR = ROOT / "paper/manuscript/figures"

CLASS_NAMES = {
    0: "R-EH",
    1: "S-EH",
    2: "T-EH",
    3: "Full",
    4: "R-Tr",
    5: "S-Tr",
    6: "T-Tr",
    7: "Off",
    8: "Idle",
    9: "Unload",
    10: "Road",
}

CLASS_GROUPS = {
    0: "Harvesting",
    1: "Harvesting",
    2: "Harvesting",
    3: "Harvesting",
    4: "Transfer",
    5: "Transfer",
    6: "Transfer",
    7: "Waiting",
    8: "Waiting",
    9: "Unloading",
    10: "Road",
}

GROUP_COLORS = {
    "Harvesting": "#4E79A7",
    "Transfer": "#F28E2B",
    "Waiting": "#59A14F",
    "Unloading": "#B07AA1",
    "Road": "#E15759",
}


def setup_style():
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="DejaVu Sans",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 320,
        },
    )


def read_class_counts():
    rows = []
    for split in ["train", "val", "test"]:
        counter = Counter()
        with (DATA_DIR / f"{split}.csv").open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                counter[int(row["分类"])] += 1
        for class_id, count in sorted(counter.items()):
            rows.append(
                {
                    "Split": split.capitalize(),
                    "Class": CLASS_NAMES[class_id],
                    "Group": CLASS_GROUPS[class_id],
                    "Class ID": class_id,
                    "Samples": count,
                }
            )
    return rows


def read_run_metrics():
    rows = []
    model_name = {
        "trnet": "TRNet",
        "image_best": "Image",
        "ast": "AST",
        "trimodal_concat": "TIM concat",
    }
    with (RESULT_DIR / "paper_run_table.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            model = model_name[row["exp_id"]]
            rows.extend(
                [
                    {"Model": model, "Metric": "Accuracy", "Score": float(row["test_acc"]) * 100},
                    {"Model": model, "Metric": "Macro-F1", "Score": float(row["test_macro_f1"]) * 100},
                    {"Model": model, "Metric": "Weighted-F1", "Score": float(row["test_weighted_f1"]) * 100},
                ]
            )
    order = {"TRNet": 0, "Image": 1, "AST": 2, "TIM concat": 3}
    return sorted(rows, key=lambda r: (order[r["Model"]], r["Metric"]))


def read_per_class_f1():
    model_name = {
        "trnet": "TRNet",
        "image_best": "Image",
        "ast": "AST",
        "trimodal_concat": "TIM concat",
    }
    by_class = defaultdict(dict)
    with (RESULT_DIR / "paper_per_class_runs.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            class_id = int(row["class_id"])
            model = model_name[row["exp_id"]]
            by_class[class_id][model] = float(row["f1"]) * 100
    return by_class


def plot_distribution(rows):
    df = pd.DataFrame(rows)
    class_order = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]
    split_order = ["Train", "Val", "Test"]

    totals = (
        df.groupby(["Class ID", "Class", "Group"], as_index=False)["Samples"]
        .sum()
        .sort_values("Class ID")
    )
    split_totals = df.groupby("Split")["Samples"].transform("sum")
    df["Share"] = df["Samples"] / split_totals * 100.0
    share_table = (
        df.pivot(index="Split", columns="Class", values="Share")
        .reindex(split_order)
        .loc[:, class_order]
    )

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(7.4, 5.2),
        gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.42},
    )

    colors = [GROUP_COLORS[group] for group in totals["Group"]]
    x = list(range(len(totals)))
    ax1.bar(x, totals["Samples"], color=colors, edgecolor="#333333", linewidth=0.35)
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(totals["Class"].tolist())
    ax1.set_xlabel("Operation class")
    ax1.set_ylabel("Total samples (log scale)")
    ax1.grid(False)
    ax1.xaxis.grid(False, which="both")
    ax1.yaxis.grid(False, which="both")
    ax1.set_ylim(100, max(totals["Samples"]) * 2.0)
    for xpos, count in zip(x, totals["Samples"]):
        label = f"{count / 1000:.1f}k" if count >= 10000 else f"{int(count):,}"
        ax1.text(xpos, count * 1.12, label, ha="center", va="bottom", fontsize=7)
    handles = [Patch(facecolor=color, edgecolor="#333333", label=group) for group, color in GROUP_COLORS.items()]
    ax1.legend(
        handles=handles,
        title=None,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        frameon=True,
    )

    sns.heatmap(
        share_table,
        ax=ax2,
        cmap="YlGnBu",
        annot=True,
        fmt=".1f",
        linewidths=0.45,
        linecolor="white",
        cbar_kws={"label": "Share within split (%)"},
        annot_kws={"fontsize": 7},
        vmin=0,
        vmax=42,
    )
    ax2.set_xlabel("Operation class")
    ax2.set_ylabel("Data split")
    ax2.tick_params(axis="x", rotation=0)
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.08, right=0.92, hspace=0.55)
    fig.savefig(FIG_DIR / "fig_dataset_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_overall_metrics(rows):
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    palette = ["#4C78A8", "#F58518", "#54A24B"]
    sns.barplot(data=pd.DataFrame(rows), x="Model", y="Score", hue="Metric", ax=ax, palette=palette)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 90)
    ax.legend(title=None, loc="upper left", frameon=True)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=2, fontsize=7)
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.35)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_overall_results.png", bbox_inches="tight")
    plt.close(fig)


def plot_modality_roles(by_class):
    models = ["TRNet", "Image", "AST", "TIM concat"]
    class_ids = sorted(by_class)
    matrix = [[by_class[c][m] for c in class_ids] for m in models]
    deltas = []
    colors = []
    for c in class_ids:
        best_single = max(by_class[c][m] for m in ["TRNet", "Image", "AST"])
        delta = by_class[c]["TIM concat"] - best_single
        deltas.append(delta)
        colors.append("#54A24B" if delta >= 0 else "#D62728")

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(7.4, 5.3),
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.38},
    )
    sns.heatmap(
        matrix,
        ax=ax1,
        cmap="YlGnBu",
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "F1 (%)"},
        xticklabels=[CLASS_NAMES[c] for c in class_ids],
        yticklabels=models,
        vmin=0,
        vmax=100,
        annot_kws={"fontsize": 7},
    )
    ax1.set_xlabel("Operation class")
    ax1.set_ylabel("Model")

    ax2.bar([CLASS_NAMES[c] for c in class_ids], deltas, color=colors, edgecolor="#333333", linewidth=0.4)
    ax2.axhline(0, color="#333333", linewidth=0.8)
    ax2.set_xlabel("Operation class")
    ax2.set_ylabel("TIM - best\nsingle F1 (pp)")
    ax2.grid(True, axis="y", linewidth=0.5, alpha=0.35)
    for tick in ax2.get_xticklabels():
        tick.set_rotation(0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_modality_roles.png", bbox_inches="tight")
    plt.close(fig)


def main():
    setup_style()
    plot_distribution(read_class_counts())
    plot_overall_metrics(read_run_metrics())
    plot_modality_roles(read_per_class_f1())


if __name__ == "__main__":
    main()
