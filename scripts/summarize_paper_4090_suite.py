#!/usr/bin/env python3
"""Summarize paper-scale 4090 experiment suites."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import pandas as pd


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


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.fmean(values)), float(statistics.stdev(values))


def collect_runs(suite_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for summary_path in sorted(suite_dir.glob("*/seed*/summary.json")):
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        args = summary.get("args", {})
        test = summary.get("test", {})
        exp_id = summary_path.parent.parent.name
        seed_text = summary_path.parent.name.replace("seed", "")
        rows.append({
            "exp_id": exp_id,
            "seed": int(args.get("seed", seed_text or 0)),
            "run_dir": str(summary_path.parent.relative_to(PROJECT_ROOT)),
            "mode": args.get("mode", ""),
            "fusion": args.get("fusion", ""),
            "seq_len": int(args.get("seq_len", 0)),
            "batch_size": int(args.get("batch_size", 0)),
            "epochs": int(args.get("epochs", 0)),
            "best_val_macro_f1": float(summary.get("best_val_macro_f1", 0.0)),
            "test_loss": float(test.get("loss", 0.0)),
            "test_acc": float(test.get("acc", 0.0)),
            "test_macro_f1": float(test.get("macro_f1", 0.0)),
            "test_weighted_f1": float(test.get("weighted_f1", 0.0)),
        })
    rows.sort(key=lambda row: (row["exp_id"], row["seed"]))
    return rows


def write_run_table(rows: list[dict], output_dir: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with (output_dir / "paper_run_table.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_runs(rows: list[dict]) -> list[dict]:
    aggregate: list[dict] = []
    for exp_id in sorted({row["exp_id"] for row in rows}):
        group = [row for row in rows if row["exp_id"] == exp_id]
        row = {
            "exp_id": exp_id,
            "num_runs": len(group),
        }
        for key in ["best_val_macro_f1", "test_acc", "test_macro_f1", "test_weighted_f1", "test_loss"]:
            values = [float(item[key]) for item in group]
            mean, std = mean_std(values)
            row[f"{key}_mean"] = mean
            row[f"{key}_std"] = std
            row[f"{key}_min"] = min(values) if values else 0.0
            row[f"{key}_max"] = max(values) if values else 0.0
        best = max(group, key=lambda item: item["test_macro_f1"])
        row["best_seed"] = best["seed"]
        row["best_run_dir"] = best["run_dir"]
        aggregate.append(row)
    aggregate.sort(key=lambda row: row["test_macro_f1_mean"], reverse=True)
    return aggregate


def write_aggregate(rows: list[dict], output_dir: Path) -> list[dict]:
    aggregate = aggregate_runs(rows)
    if not aggregate:
        return []
    fieldnames = list(aggregate[0].keys())
    with (output_dir / "paper_aggregate.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate)
    return aggregate


def collect_per_class(suite_dir: Path, output_dir: Path) -> None:
    rows: list[dict] = []
    for metrics_path in sorted(suite_dir.glob("*/seed*/per_class_metrics.csv")):
        exp_id = metrics_path.parent.parent.name
        summary_path = metrics_path.parent / "summary.json"
        seed = 0
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as f:
                seed = int(json.load(f).get("args", {}).get("seed", 0))
        metrics = pd.read_csv(metrics_path)
        for _, item in metrics.iterrows():
            rows.append({
                "exp_id": exp_id,
                "seed": seed,
                "class_id": int(item["class_id"]),
                "class_name": item.get("class_name", CLASS_NAMES[int(item["class_id"])]),
                "precision": float(item.get("precision", 0.0)),
                "recall": float(item.get("recall", 0.0)),
                "f1": float(item.get("f1", 0.0)),
                "support": int(item.get("support", 0)),
            })
    if not rows:
        return
    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / "paper_per_class_runs.csv", index=False, encoding="utf-8-sig")
    agg = (
        raw.groupby(["exp_id", "class_id", "class_name"], as_index=False)
        .agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            support_mean=("support", "mean"),
        )
        .fillna(0.0)
    )
    agg.to_csv(output_dir / "paper_per_class_aggregate.csv", index=False, encoding="utf-8-sig")


def collect_part_diagnostics(suite_dir: Path, output_dir: Path) -> None:
    rows: list[dict] = []
    for path in sorted(suite_dir.glob("*/seed*/part_diagnostics/**/*_part_diagnostic_summary.csv")):
        if path.name == "part_diagnostic_summary.csv":
            continue
        exp_id = path.parents[3].name
        seed_dir = path.parents[2].name
        date = path.parent.name
        try:
            seed = int(seed_dir.replace("seed", ""))
        except ValueError:
            seed = 0
        data = pd.read_csv(path)
        for _, item in data.iterrows():
            rows.append({
                "exp_id": exp_id,
                "seed": seed,
                "date": date,
                "part": int(item["part"]),
                "rows": int(item["rows"]),
                "acc": float(item["acc"]),
                "wrong": int(item["wrong"]),
                "top_errors": item.get("top_errors", ""),
            })
    if not rows:
        return
    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / "paper_part_diagnostics.csv", index=False, encoding="utf-8-sig")
    agg = (
        raw.groupby(["exp_id", "date", "part"], as_index=False)
        .agg(acc_mean=("acc", "mean"), acc_std=("acc", "std"), wrong_mean=("wrong", "mean"), rows_mean=("rows", "mean"))
        .fillna(0.0)
    )
    agg.to_csv(output_dir / "paper_part_diagnostics_aggregate.csv", index=False, encoding="utf-8-sig")


def write_markdown(rows: list[dict], aggregate: list[dict], output_dir: Path) -> None:
    lines = ["# Paper 4090 Suite Summary", ""]
    if aggregate:
        lines.extend([
            "## Aggregate",
            "",
            "| rank | exp_id | runs | macro-F1 mean±std | acc mean±std | weighted-F1 mean±std | best seed |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ])
        for rank, row in enumerate(aggregate, start=1):
            lines.append(
                f"| {rank} | {row['exp_id']} | {row['num_runs']} | "
                f"{row['test_macro_f1_mean']:.4f}±{row['test_macro_f1_std']:.4f} | "
                f"{row['test_acc_mean']:.4f}±{row['test_acc_std']:.4f} | "
                f"{row['test_weighted_f1_mean']:.4f}±{row['test_weighted_f1_std']:.4f} | "
                f"{row['best_seed']} |"
            )
    if rows:
        lines.extend([
            "",
            "## Runs",
            "",
            "| exp_id | seed | test macro-F1 | test acc | weighted-F1 | best val macro-F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in rows:
            lines.append(
                f"| {row['exp_id']} | {row['seed']} | {row['test_macro_f1']:.4f} | "
                f"{row['test_acc']:.4f} | {row['test_weighted_f1']:.4f} | {row['best_val_macro_f1']:.4f} |"
            )
    (output_dir / "paper_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", default="experiments/paper_4090_20260417")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    suite_dir = resolve(args.suite_dir)
    output_dir = resolve(args.output_dir) if args.output_dir else suite_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_runs(suite_dir)
    write_run_table(rows, output_dir)
    aggregate = write_aggregate(rows, output_dir)
    collect_per_class(suite_dir, output_dir)
    collect_part_diagnostics(suite_dir, output_dir)
    write_markdown(rows, aggregate, output_dir)
    print(f"Wrote summary to {output_dir}")


if __name__ == "__main__":
    main()
