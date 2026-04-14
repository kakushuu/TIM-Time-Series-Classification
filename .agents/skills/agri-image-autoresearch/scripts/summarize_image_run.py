#!/usr/bin/env python3
"""Emit parseable metrics from an Agri-MBT image-only summary.json."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="Path to summary.json from an image-only run")
    parser.add_argument("--per-class", default="", help="Optional path to per_class_metrics.csv")
    parser.add_argument("--json", action="store_true", help="Emit one JSON object instead of METRIC lines")
    return parser.parse_args()


TARGET_CLASSES = {1, 2, 4, 5, 6, 8, 9}
TARGET_THRESHOLD = 0.30


def load_per_class(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    f1_values: list[float] = []
    recall_values: list[float] = []
    target_f1_values: list[float] = []
    target_recall_values: list[float] = []
    worst_f1 = 1.0
    worst_recall = 1.0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            class_id = int(row.get("class_id", -1))
            f1 = float(row.get("f1", 0.0) or 0.0)
            recall = float(row.get("recall", 0.0) or 0.0)
            f1_values.append(f1)
            recall_values.append(recall)
            worst_f1 = min(worst_f1, f1)
            worst_recall = min(worst_recall, recall)
            if class_id in TARGET_CLASSES:
                target_f1_values.append(f1)
                target_recall_values.append(recall)
    if not f1_values:
        return {}
    metrics = {
        "per_class_f1_mean": mean(f1_values),
        "per_class_recall_mean": mean(recall_values),
        "worst_class_f1": worst_f1,
        "worst_class_recall": worst_recall,
    }
    if target_f1_values:
        metrics.update({
            "target_class_f1_min": min(target_f1_values),
            "target_class_recall_min": min(target_recall_values),
            "target_class_f1_mean": mean(target_f1_values),
            "target_class_recall_mean": mean(target_recall_values),
            "target_classes_f1_ge_030": sum(v >= TARGET_THRESHOLD for v in target_f1_values),
            "target_classes_recall_ge_030": sum(v >= TARGET_THRESHOLD for v in target_recall_values),
        })
    return metrics


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    if summary.get("mode") != "image_only":
        raise SystemExit(f"Expected image_only summary, got mode={summary.get('mode')!r}")

    history = summary.get("history", [])
    final = history[-1] if history else {"train": {}, "val": {}}
    final_train = final.get("train", {})
    final_val = final.get("val", {})
    test = summary.get("test", {})

    metrics = {
        "best_val_macro_f1": float(summary.get("best_val_macro_f1", 0.0)),
        "final_train_macro_f1": float(final_train.get("macro_f1", 0.0)),
        "final_val_macro_f1": float(final_val.get("macro_f1", 0.0)),
        "final_train_val_macro_f1_gap": float(final_train.get("macro_f1", 0.0)) - float(final_val.get("macro_f1", 0.0)),
        "test_macro_f1": float(test.get("macro_f1", 0.0)),
        "test_weighted_f1": float(test.get("weighted_f1", 0.0)),
        "test_acc": float(test.get("acc", 0.0)),
        "test_loss": float(test.get("loss", 0.0)),
    }

    per_class_path = Path(args.per_class) if args.per_class else summary_path.with_name("per_class_metrics.csv")
    metrics.update(load_per_class(per_class_path))

    if args.json:
        print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
        return

    for name, value in metrics.items():
        print(f"METRIC {name}={value:.10f}")


if __name__ == "__main__":
    main()
