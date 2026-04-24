#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ACTION="${1:-start}"
SUFFIX="${2:-${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}}"
GPU_IDS="${TRNET_SEQ_GPU_IDS:-0,1,2,5}"
SEEDS="${TRNET_SEQ_SEEDS:-42 43 44 45}"
SPLIT_DIR="${TRNET_SEQ_SPLIT_DIR:-data/b_deep_part_multimodal_full_clean_20260417}"
TRAIN_CSV="${TRNET_SEQ_TRAIN_CSV:-$SPLIT_DIR/train.csv}"
VAL_CSV="${TRNET_SEQ_VAL_CSV:-$SPLIT_DIR/val.csv}"
TEST_CSV="${TRNET_SEQ_TEST_CSV:-$SPLIT_DIR/test.csv}"
DURATION_STATS="${TRNET_SEQ_DURATION_STATS:-experiments/b_deep_part_duration_analysis/duration_sampling_config.json}"
SUITE_DIR="${TRNET_SEQ_SUITE_DIR:-experiments/trnet_seq_4090}"
LOG_DIR="${TRNET_SEQ_LOG_DIR:-logs/trnet_seq_4090_${SUFFIX}}"
PID_FILE="$LOG_DIR/pid"
CONDA_BIN="${TRNET_SEQ_CONDA_BIN:-/private/miniforge3/bin/conda}"
CONDA_ENV="${TRNET_SEQ_CONDA_ENV:-agri-mbt}"
EPOCHS="${TRNET_SEQ_EPOCHS:-12}"
BATCH_SIZE="${TRNET_SEQ_BATCH_SIZE:-8}"
NUM_WORKERS="${TRNET_SEQ_NUM_WORKERS:-4}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_trnet_seq_4090_experiment.sh smoke [RUN_SUFFIX]
  scripts/run_trnet_seq_4090_experiment.sh start [RUN_SUFFIX]
  scripts/run_trnet_seq_4090_experiment.sh run RUN_SUFFIX
  scripts/run_trnet_seq_4090_experiment.sh status [RUN_SUFFIX]
  scripts/run_trnet_seq_4090_experiment.sh summarize [RUN_SUFFIX]

Environment overrides:
  TRNET_SEQ_GPU_IDS       Physical GPU ids. Default: 0,1,2,5
  TRNET_SEQ_SEEDS         One seed per parallel run. Default: 42 43 44 45
  TRNET_SEQ_EPOCHS        Default: 12
  TRNET_SEQ_NUM_WORKERS   Default: 4
USAGE
}

require_4090_gpus() {
  local gpu_table id name
  gpu_table="$(nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader)"
  IFS=',' read -r -a ids <<<"$GPU_IDS"
  for id in "${ids[@]}"; do
    id="${id//[[:space:]]/}"
    name="$(awk -F',' -v id="$id" '$1 + 0 == id {gsub(/^[ \t]+|[ \t]+$/, "", $3); print $3}' <<<"$gpu_table")"
    if [[ -z "$name" ]]; then
      echo "GPU $id not found" >&2
      exit 1
    fi
    if [[ "$name" != *"4090"* ]]; then
      echo "GPU $id is '$name', not an RTX 4090" >&2
      exit 1
    fi
  done
}

check_inputs() {
  require_4090_gpus
  for path in "$TRAIN_CSV" "$VAL_CSV" "$TEST_CSV" "$DURATION_STATS"; do
    if [[ ! -f "$path" ]]; then
      echo "Missing required input: $path" >&2
      exit 1
    fi
    echo "Found: $path"
  done
}

run_id_for() {
  local seed="$1"
  printf 'trnet_seq_seed%s_%s' "$seed" "$SUFFIX"
}

train_one() {
  local gpu="$1"
  local seed="$2"
  local run_id
  run_id="$(run_id_for "$seed")"
  mkdir -p "$SUITE_DIR/$run_id"
  echo "[$(date -Is)] START gpu=$gpu seed=$seed run_id=$run_id"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
    python src/train_ablation.py \
      --mode trajectory_only \
      --train-csv "$TRAIN_CSV" \
      --val-csv "$VAL_CSV" \
      --test-csv "$TEST_CSV" \
      --save-dir "$SUITE_DIR/$run_id" \
      --seq-len 512 \
      --stride 20 \
      --eval-stride 1 \
      --context-mode causal \
      --sampling-strategy adaptive \
      --duration-stats "$DURATION_STATS" \
      --feature-mode engineered \
      --traj-encoder trnet_seq \
      --traj-feature-map-size 6 \
      --max-time-gap 1 \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --device cuda \
      --seed "$seed" \
      --early-stop-patience 4 \
      --early-stop-min-delta 0.001
  echo "[$(date -Is)] DONE gpu=$gpu seed=$seed run_id=$run_id"
}

smoke() {
  check_inputs
  mkdir -p "$LOG_DIR"
  local gpu="${GPU_IDS%%,*}"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
    python src/train_ablation.py \
      --mode trajectory_only \
      --train-csv "$TRAIN_CSV" \
      --val-csv "$VAL_CSV" \
      --test-csv "$TEST_CSV" \
      --save-dir "$SUITE_DIR/trnet_seq_gpu_smoke_${SUFFIX}" \
      --seq-len 512 \
      --stride 200 \
      --eval-stride 500 \
      --context-mode causal \
      --sampling-strategy fixed \
      --feature-mode engineered \
      --traj-encoder trnet_seq \
      --traj-feature-map-size 6 \
      --max-time-gap 1 \
      --epochs 1 \
      --batch-size 4 \
      --num-workers 0 \
      --device cuda \
      --max-train-batches 2 \
      --max-eval-batches 2
}

summarize() {
  mkdir -p "$LOG_DIR/results"
  python - "$SUITE_DIR" "$LOG_DIR/results" "$SUFFIX" <<'PY'
import csv
import json
import sys
from pathlib import Path

suite_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
suffix = sys.argv[3]
rows = []
for summary_path in sorted(suite_dir.glob(f"trnet_seq_seed*_{suffix}/summary.json")):
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    test = summary.get("test", {})
    args = summary.get("args", {})
    rows.append({
        "run_id": summary_path.parent.name,
        "seed": args.get("seed", ""),
        "best_val_macro_f1": summary.get("best_val_macro_f1", 0.0),
        "test_acc": test.get("acc", 0.0),
        "test_macro_f1": test.get("macro_f1", 0.0),
        "test_weighted_f1": test.get("weighted_f1", 0.0),
        "test_loss": test.get("loss", 0.0),
    })
csv_path = output_dir / "trnet_seq_4090_summary.csv"
with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["run_id", "seed", "best_val_macro_f1", "test_acc", "test_macro_f1", "test_weighted_f1", "test_loss"])
    writer.writeheader()
    writer.writerows(rows)
aggregate = {"num_runs": len(rows)}
for key in ["best_val_macro_f1", "test_acc", "test_macro_f1", "test_weighted_f1", "test_loss"]:
    values = [float(row[key]) for row in rows]
    if values:
        aggregate[f"{key}_mean"] = sum(values) / len(values)
        aggregate[f"{key}_min"] = min(values)
        aggregate[f"{key}_max"] = max(values)
with (output_dir / "trnet_seq_4090_summary.json").open("w", encoding="utf-8") as f:
    json.dump({"runs": rows, "aggregate": aggregate}, f, ensure_ascii=False, indent=2)
print(f"Wrote {csv_path}")
print(json.dumps(aggregate, ensure_ascii=False, sort_keys=True))
PY
}

run_all() {
  check_inputs
  mkdir -p "$LOG_DIR"
  echo "suffix=$SUFFIX" > "$LOG_DIR/config.env"
  echo "gpu_ids=$GPU_IDS" >> "$LOG_DIR/config.env"
  echo "seeds=$SEEDS" >> "$LOG_DIR/config.env"
  echo "epochs=$EPOCHS" >> "$LOG_DIR/config.env"
  IFS=',' read -r -a gpus <<<"$GPU_IDS"
  read -r -a seeds <<<"$SEEDS"
  if [[ "${#gpus[@]}" -ne "${#seeds[@]}" ]]; then
    echo "TRNET_SEQ_GPU_IDS and TRNET_SEQ_SEEDS must have the same count" >&2
    exit 2
  fi
  local -a pids=()
  local idx
  for idx in "${!gpus[@]}"; do
    local gpu="${gpus[$idx]//[[:space:]]/}"
    local seed="${seeds[$idx]}"
    local run_id
    run_id="$(run_id_for "$seed")"
    train_one "$gpu" "$seed" > "$LOG_DIR/${run_id}.log" 2>&1 &
    pids+=("$!")
    echo "$!" > "$LOG_DIR/${run_id}.pid"
  done
  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  summarize || true
  exit "$status"
}

case "$ACTION" in
  smoke)
    smoke
    ;;
  start)
    mkdir -p "$LOG_DIR"
    nohup "$0" run "$SUFFIX" > "$LOG_DIR/run.log" 2>&1 &
    echo "$!" > "$PID_FILE"
    echo "Started TRNet sequence 4090 experiment."
    echo "PID: $(cat "$PID_FILE")"
    echo "Log: $LOG_DIR/run.log"
    ;;
  run)
    run_all
    ;;
  status)
    if [[ -f "$PID_FILE" ]] && ps -p "$(cat "$PID_FILE")" >/dev/null 2>&1; then
      echo "running pid=$(cat "$PID_FILE")"
    elif [[ -f "$PID_FILE" ]]; then
      echo "not running pid=$(cat "$PID_FILE")"
    else
      echo "no pid file: $PID_FILE"
    fi
    ;;
  summarize)
    summarize
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    usage >&2
    exit 2
    ;;
esac
