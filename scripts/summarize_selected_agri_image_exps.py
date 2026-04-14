#!/usr/bin/env python3
"""Summarize selected Agri image experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_CLASSES = {1, 2, 4, 5, 6, 8, 9}


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def best_epoch(summary: dict) -> int | None:
    history = summary.get("history") or []
    if not history:
        return None
    best = max(history, key=lambda item: item.get("val", {}).get("macro_f1", -1.0))
    return int(best.get("epoch", 0))


def load_per_class(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def float_or_blank(value) -> float | str:
    if value is None:
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def summarize_run(row: dict[str, str], suite_dir: Path) -> dict[str, object]:
    run_id = row["run_id"]
    run_dir = suite_dir / run_id
    summary_path = run_dir / "summary.json"
    per_class_path = run_dir / "per_class_metrics.csv"
    result: dict[str, object] = {
        "exp_id": row["exp_id"],
        "run_id": run_id,
        "description": row.get("description", ""),
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)) if run_dir.is_relative_to(PROJECT_ROOT) else str(run_dir),
    }
    if not summary_path.exists():
        result.update({"status": "missing_summary"})
        return result

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    test = summary.get("test", {})
    result.update(
        {
            "status": "ok",
            "best_epoch": best_epoch(summary),
            "best_val_macro_f1": float_or_blank(summary.get("best_val_macro_f1")),
            "test_macro_f1": float_or_blank(test.get("macro_f1")),
            "test_weighted_f1": float_or_blank(test.get("weighted_f1")),
            "test_acc": float_or_blank(test.get("acc")),
            "test_loss": float_or_blank(test.get("loss")),
        }
    )

    per_class = load_per_class(per_class_path)
    if per_class:
        f1s = [float(r["f1"]) for r in per_class if r.get("f1") not in (None, "")]
        recalls = [float(r["recall"]) for r in per_class if r.get("recall") not in (None, "")]
        target = [r for r in per_class if int(float(r["class_id"])) in TARGET_CLASSES]
        target_f1s = [float(r["f1"]) for r in target]
        target_recalls = [float(r["recall"]) for r in target]
        weakest = sorted(per_class, key=lambda r: float(r["f1"]))[:3]
        result.update(
            {
                "worst_class_f1": min(f1s) if f1s else "",
                "worst_class_recall": min(recalls) if recalls else "",
                "target_class_f1_mean": mean(target_f1s) if target_f1s else "",
                "target_class_f1_min": min(target_f1s) if target_f1s else "",
                "target_class_recall_mean": mean(target_recalls) if target_recalls else "",
                "target_class_recall_min": min(target_recalls) if target_recalls else "",
                "target_classes_f1_ge_030": sum(v >= 0.30 for v in target_f1s),
                "target_classes_recall_ge_030": sum(v >= 0.30 for v in target_recalls),
                "weakest_classes": "; ".join(
                    f"class {int(float(r['class_id']))}: f1={float(r['f1']):.4f}, recall={float(r['recall']):.4f}, support={int(float(r['support']))}"
                    for r in weakest
                ),
            }
        )
    return result


def fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return ""
    return str(value)


def write_markdown(rows: list[dict[str, object]], output_path: Path) -> None:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    by_val = sorted(ok_rows, key=lambda r: float(r.get("best_val_macro_f1") or -1), reverse=True)
    by_test = sorted(ok_rows, key=lambda r: float(r.get("test_macro_f1") or -1), reverse=True)
    cols = [
        "exp_id",
        "run_id",
        "best_epoch",
        "best_val_macro_f1",
        "test_macro_f1",
        "test_weighted_f1",
        "test_acc",
        "target_class_f1_min",
        "target_classes_f1_ge_030",
    ]
    lines = ["# Selected Agri Image Experiment Summary", ""]
    if by_val:
        lines.append(f"Best validation macro F1: **{by_val[0]['run_id']}** ({fmt(by_val[0].get('best_val_macro_f1'))}).")
    if by_test:
        lines.append(f"Best test macro F1: **{by_test[0]['run_id']}** ({fmt(by_test[0].get('test_macro_f1'))}).")
    lines.extend(["", "| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"])
    for row in by_val:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in cols) + " |")
    lines.extend(["", "## Weakest Classes", ""])
    for row in by_val:
        lines.append(f"- `{row['run_id']}`: {row.get('weakest_classes', '')}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--suite-dir", default="experiments/agri_image_autoresearch")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    manifest = resolve(args.manifest)
    suite_dir = resolve(args.suite_dir)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [summarize_run(row, suite_dir) for row in load_manifest(manifest)]
    fieldnames = [
        "status",
        "exp_id",
        "run_id",
        "description",
        "best_epoch",
        "best_val_macro_f1",
        "test_macro_f1",
        "test_weighted_f1",
        "test_acc",
        "test_loss",
        "worst_class_f1",
        "worst_class_recall",
        "target_class_f1_mean",
        "target_class_f1_min",
        "target_class_recall_mean",
        "target_class_recall_min",
        "target_classes_f1_ge_030",
        "target_classes_recall_ge_030",
        "weakest_classes",
        "run_dir",
    ]
    with (output_dir / "results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)
    write_markdown(rows, output_dir / "summary.md")
    print(f"Wrote {output_dir / 'results.csv'}")
    print(f"Wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
