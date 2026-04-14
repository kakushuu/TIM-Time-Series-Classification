#!/usr/bin/env python3
"""Create Agri-MBT image-only autoresearch session files."""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-ids", default="1,2,5,6")
    parser.add_argument("--suite-dir", default="experiments/agri_image_autoresearch")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--eval-stride", type=int, default=1)
    parser.add_argument("--image-window-size", type=int, default=9)
    parser.add_argument("--image-radius", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--conda-bin", default="/private/miniforge3/bin/conda")
    parser.add_argument("--conda-env", default="agri-mbt")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    return parser.parse_args()


def write_file(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        print(f"skip existing: {path}")
        return
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111 if path.suffix == ".sh" else path.stat().st_mode)
    print(f"wrote: {path}")


def runner_content(args: argparse.Namespace) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GPU_IDS="${{AGRI_IMAGE_GPU_IDS:-{args.gpu_ids}}}"
CONDA_BIN="${{AGRI_IMAGE_CONDA_BIN:-{args.conda_bin}}}"
CONDA_ENV="${{AGRI_IMAGE_CONDA_ENV:-{args.conda_env}}}"
RUN_ID="${{AGRI_IMAGE_RUN_ID:-$(date +%Y%m%d_%H%M%S)}}"
SUITE_DIR="${{AGRI_IMAGE_SUITE_DIR:-{args.suite_dir}}}"
SAVE_DIR="$SUITE_DIR/$RUN_ID"

TRAIN_CSV="${{AGRI_IMAGE_TRAIN_CSV:-data/taif_20241018_split/train.csv}}"
VAL_CSV="${{AGRI_IMAGE_VAL_CSV:-data/taif_20241018_split/val.csv}}"
TEST_CSV="${{AGRI_IMAGE_TEST_CSV:-data/taif_20241018_split/test.csv}}"
DURATION_STATS="${{AGRI_IMAGE_DURATION_STATS:-experiments/new_adaptive_mbt_20241018_full/behavior_duration_analysis/duration_sampling_config.json}}"

EPOCHS="${{AGRI_IMAGE_EPOCHS:-{args.epochs}}}"
BATCH_SIZE="${{AGRI_IMAGE_BATCH_SIZE:-{args.batch_size}}}"
NUM_WORKERS="${{AGRI_IMAGE_NUM_WORKERS:-{args.num_workers}}}"
SEQ_LEN="${{AGRI_IMAGE_SEQ_LEN:-{args.seq_len}}}"
STRIDE="${{AGRI_IMAGE_STRIDE:-{args.stride}}}"
EVAL_STRIDE="${{AGRI_IMAGE_EVAL_STRIDE:-{args.eval_stride}}}"
IMAGE_WINDOW_SIZE="${{AGRI_IMAGE_WINDOW_SIZE:-{args.image_window_size}}}"
IMAGE_SAMPLING="${{AGRI_IMAGE_SAMPLING:-center}}"
IMAGE_RADIUS="${{AGRI_IMAGE_RADIUS:-{args.image_radius}}}"
LR="${{AGRI_IMAGE_LR:-{args.lr}}}"
WEIGHT_DECAY="${{AGRI_IMAGE_WEIGHT_DECAY:-{args.weight_decay}}}"
MAX_TRAIN_BATCHES="${{AGRI_IMAGE_MAX_TRAIN_BATCHES:-0}}"
MAX_EVAL_BATCHES="${{AGRI_IMAGE_MAX_EVAL_BATCHES:-0}}"

mkdir -p "$SAVE_DIR"

CUDA_VISIBLE_DEVICES="$GPU_IDS" "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \\
  python src/train_ablation.py \\
    --mode image_only \\
    --train-csv "$TRAIN_CSV" \\
    --val-csv "$VAL_CSV" \\
    --test-csv "$TEST_CSV" \\
    --save-dir "$SAVE_DIR" \\
    --seq-len "$SEQ_LEN" \\
    --stride "$STRIDE" \\
    --eval-stride "$EVAL_STRIDE" \\
    --context-mode causal \\
    --sampling-strategy adaptive \\
    --duration-stats "$DURATION_STATS" \\
    --image-window-size "$IMAGE_WINDOW_SIZE" \\
    --image-sampling "$IMAGE_SAMPLING" \\
    --image-radius "$IMAGE_RADIUS" \\
    --epochs "$EPOCHS" \\
    --lr "$LR" \\
    --weight-decay "$WEIGHT_DECAY" \\
    --batch-size "$BATCH_SIZE" \\
    --num-workers "$NUM_WORKERS" \\
    --device cuda \\
    --all-gpus \\
    --pretrained \\
    --max-train-batches "$MAX_TRAIN_BATCHES" \\
    --max-eval-batches "$MAX_EVAL_BATCHES"

python .agents/skills/agri-image-autoresearch/scripts/summarize_image_run.py \\
  --summary "$SAVE_DIR/summary.json"
"""


def checks_content() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python -m py_compile src/train_ablation.py src/models/visual_encoder.py

if [[ ! -f data/taif_20241018_split/train.csv ]]; then
  echo "missing train split" >&2
  exit 1
fi
"""


def brief_content(args: argparse.Namespace) -> str:
    return f"""# Autoresearch: Agri-MBT Image Only

## Objective

Improve image-only validation macro F1 before trajectory-only retuning and multimodal fusion experiments.

## Up-Front Answers

- Primary metric: `best_val_macro_f1`
- Unit: F1
- Direction: higher
- Minimum meaningful improvement: +0.01 absolute macro F1
- Workload command: `./autoresearch.sh`
- Correctness gates: `./autoresearch.checks.sh`
- Budget / stop criteria: stop after 5 coherent non-improving experiments or when image-only errors are clearly trajectory-dependent

## Standard Workload

- GPUs: `{args.gpu_ids}`
- Epochs: `{args.epochs}`
- Batch size: `{args.batch_size}`
- Train split: `data/taif_20241018_split/train.csv`
- Validation split: `data/taif_20241018_split/val.csv`
- Test split: `data/taif_20241018_split/test.csv`
- Output root: `{args.suite_dir}`

## Scope

- In scope: `src/train_ablation.py`, `src/models/visual_encoder.py`, focused helper scripts
- Off limits: raw data, split changes, trajectory-only and multimodal changes

## Decision Rule

Keep a candidate if `best_val_macro_f1` improves by at least +0.01 over the current best and rare-class F1/recall do not collapse. Test metrics are audit-only.

## Experiment Ledger

`.autoresearch/results.jsonl`

## Report Outputs

- `.autoresearch/report.html`
- `.autoresearch/results.csv`

## Current Best Result

Not established yet.

## What We've Learned

- Historical image-only run under `experiments/new_adaptive_mbt_20241018_full/image_only` had validation macro F1 around 0.43 and test macro F1 around 0.36.
- Prioritize data sanity, augmentation, imbalance handling, image sampling, frame pooling, and optimizer schedule before multimodal work.
"""


def ensure_exclude() -> None:
    exclude = PROJECT_ROOT / ".git" / "info" / "exclude"
    if not exclude.exists():
        return
    text = exclude.read_text(encoding="utf-8", errors="ignore")
    if ".autoresearch/" not in text:
        exclude.write_text(text.rstrip() + "\n.autoresearch/\n", encoding="utf-8")
        print(f"updated: {exclude}")


def main() -> None:
    args = parse_args()
    write_file(PROJECT_ROOT / "autoresearch.sh", runner_content(args), args.force)
    write_file(PROJECT_ROOT / "autoresearch.checks.sh", checks_content(), args.force)
    write_file(PROJECT_ROOT / "autoresearch.md", brief_content(args), args.force)
    (PROJECT_ROOT / ".autoresearch").mkdir(exist_ok=True)
    ensure_exclude()


if __name__ == "__main__":
    main()
