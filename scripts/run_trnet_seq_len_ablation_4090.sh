#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ACTION="${1:-start}"
SUFFIX="${2:-${TRNET_SEQ_LEN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}}"
GPU_IDS="${TRNET_SEQ_LEN_GPU_IDS:-0,1,2,5}"
SEEDS="${TRNET_SEQ_LEN_SEEDS:-42 43 44 45}"
SEQ_LENS="${TRNET_SEQ_LEN_VALUES:-64 128 256}"
SPLIT_DIR="${TRNET_SEQ_LEN_SPLIT_DIR:-data/b_deep_part_multimodal_full_clean_20260417}"
TRAIN_CSV="${TRNET_SEQ_LEN_TRAIN_CSV:-$SPLIT_DIR/train.csv}"
VAL_CSV="${TRNET_SEQ_LEN_VAL_CSV:-$SPLIT_DIR/val.csv}"
TEST_CSV="${TRNET_SEQ_LEN_TEST_CSV:-$SPLIT_DIR/test.csv}"
DURATION_STATS="${TRNET_SEQ_LEN_DURATION_STATS:-experiments/b_deep_part_duration_analysis/duration_sampling_config.json}"
SUITE_DIR="${TRNET_SEQ_LEN_SUITE_DIR:-experiments/trnet_seq_len_ablation_4090}"
BASELINE_512_SUITE_DIR="${TRNET_SEQ_LEN_BASELINE_512_SUITE_DIR:-experiments/trnet_seq_4090}"
BASELINE_512_SUFFIX="${TRNET_SEQ_LEN_BASELINE_512_SUFFIX:-trnet_seq_20260417}"
LOG_DIR="${TRNET_SEQ_LEN_LOG_DIR:-logs/trnet_seq_len_ablation_4090_${SUFFIX}}"
PID_FILE="$LOG_DIR/pid"
CONDA_BIN="${TRNET_SEQ_LEN_CONDA_BIN:-/private/miniforge3/bin/conda}"
CONDA_ENV="${TRNET_SEQ_LEN_CONDA_ENV:-agri-mbt}"
EPOCHS="${TRNET_SEQ_LEN_EPOCHS:-12}"
BATCH_SIZE="${TRNET_SEQ_LEN_BATCH_SIZE:-8}"
NUM_WORKERS="${TRNET_SEQ_LEN_NUM_WORKERS:-4}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_trnet_seq_len_ablation_4090.sh smoke [RUN_SUFFIX]
  scripts/run_trnet_seq_len_ablation_4090.sh start [RUN_SUFFIX]
  scripts/run_trnet_seq_len_ablation_4090.sh run RUN_SUFFIX
  scripts/run_trnet_seq_len_ablation_4090.sh status [RUN_SUFFIX]
  scripts/run_trnet_seq_len_ablation_4090.sh summarize [RUN_SUFFIX]

Environment overrides:
  TRNET_SEQ_LEN_GPU_IDS       Physical GPU ids. Default: 0,1,2,5
  TRNET_SEQ_LEN_SEEDS         One seed per parallel GPU. Default: 42 43 44 45
  TRNET_SEQ_LEN_VALUES        Sequence lengths to run. Default: 64 128 256
  TRNET_SEQ_LEN_EPOCHS        Default: 12
  TRNET_SEQ_LEN_NUM_WORKERS   Default: 4
USAGE
}

require_4090_gpus() {
  local gpu_table id name attempt
  gpu_table=""
  for attempt in 1 2 3 4 5; do
    if gpu_table="$(nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader 2>&1)"; then
      break
    fi
    echo "nvidia-smi failed during GPU check (attempt $attempt/5): $gpu_table" >&2
    sleep 2
  done
  if [[ "$gpu_table" == *"NVIDIA-SMI has failed"* ]] || [[ -z "$gpu_table" ]]; then
    echo "Unable to query NVIDIA GPUs after retries: $gpu_table" >&2
    exit 1
  fi
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
  local seq_len="$1"
  local seed="$2"
  printf 'trnet_seq_len%s_seed%s_%s' "$seq_len" "$seed" "$SUFFIX"
}

train_one() {
  local gpu="$1"
  local seq_len="$2"
  local seed="$3"
  local run_id
  run_id="$(run_id_for "$seq_len" "$seed")"
  mkdir -p "$SUITE_DIR/$run_id"
  echo "[$(date -Is)] START gpu=$gpu seq_len=$seq_len seed=$seed run_id=$run_id"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
    python src/train_ablation.py \
      --mode trajectory_only \
      --train-csv "$TRAIN_CSV" \
      --val-csv "$VAL_CSV" \
      --test-csv "$TEST_CSV" \
      --save-dir "$SUITE_DIR/$run_id" \
      --seq-len "$seq_len" \
      --stride 20 \
      --eval-stride 1 \
      --context-mode causal \
      --sampling-strategy adaptive \
      --duration-stats "$DURATION_STATS" \
      --adaptive-max-window "$seq_len" \
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
  echo "[$(date -Is)] DONE gpu=$gpu seq_len=$seq_len seed=$seed run_id=$run_id"
}

smoke() {
  check_inputs
  mkdir -p "$LOG_DIR"
  local gpu="${GPU_IDS%%,*}"
  local seq_len="${SEQ_LENS%% *}"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
    python src/train_ablation.py \
      --mode trajectory_only \
      --train-csv "$TRAIN_CSV" \
      --val-csv "$VAL_CSV" \
      --test-csv "$TEST_CSV" \
      --save-dir "$SUITE_DIR/trnet_seq_len${seq_len}_gpu_smoke_${SUFFIX}" \
      --seq-len "$seq_len" \
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
      --max-eval-batches 0
}

summarize() {
  mkdir -p "$LOG_DIR/results"
  python - "$SUITE_DIR" "$LOG_DIR/results" "$SUFFIX" "$BASELINE_512_SUITE_DIR" "$BASELINE_512_SUFFIX" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

suite_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
suffix = sys.argv[3]
baseline_suite_dir = Path(sys.argv[4])
baseline_suffix = sys.argv[5]

rows = []

def add_rows(pattern, source):
    for summary_path in sorted(Path(pattern[0]).glob(pattern[1])):
        with summary_path.open("r", encoding="utf-8-sig") as f:
            summary = json.load(f)
        test = summary.get("test", {})
        args = summary.get("args", {})
        rows.append({
            "source": source,
            "run_id": summary_path.parent.name,
            "seq_len": int(args.get("seq_len", 0)),
            "seed": int(args.get("seed", 0)),
            "best_val_macro_f1": float(summary.get("best_val_macro_f1", 0.0)),
            "test_acc": float(test.get("acc", 0.0)),
            "test_macro_f1": float(test.get("macro_f1", 0.0)),
            "test_weighted_f1": float(test.get("weighted_f1", 0.0)),
            "test_loss": float(test.get("loss", 0.0)),
        })

add_rows((suite_dir, f"trnet_seq_len*_seed*_{suffix}/summary.json"), "ablation")
add_rows((baseline_suite_dir, f"trnet_seq_seed*_{baseline_suffix}/summary.json"), "baseline_512")
rows.sort(key=lambda r: (r["seq_len"], r["seed"], r["run_id"]))

fieldnames = [
    "source", "run_id", "seq_len", "seed", "best_val_macro_f1",
    "test_acc", "test_macro_f1", "test_weighted_f1", "test_loss",
]
csv_path = output_dir / "trnet_seq_len_ablation_summary.csv"
with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

aggregate = {}
for seq_len in sorted({row["seq_len"] for row in rows}):
    group = [row for row in rows if row["seq_len"] == seq_len]
    stats = {"num_runs": len(group)}
    for key in ["best_val_macro_f1", "test_acc", "test_macro_f1", "test_weighted_f1", "test_loss"]:
        values = [row[key] for row in group]
        stats[f"{key}_mean"] = statistics.fmean(values)
        stats[f"{key}_min"] = min(values)
        stats[f"{key}_max"] = max(values)
    aggregate[str(seq_len)] = stats

json_path = output_dir / "trnet_seq_len_ablation_summary.json"
with json_path.open("w", encoding="utf-8") as f:
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
  echo "seq_lens=$SEQ_LENS" >> "$LOG_DIR/config.env"
  echo "epochs=$EPOCHS" >> "$LOG_DIR/config.env"
  IFS=',' read -r -a gpus <<<"$GPU_IDS"
  read -r -a seeds <<<"$SEEDS"
  read -r -a seq_lens <<<"$SEQ_LENS"
  if [[ "${#gpus[@]}" -ne "${#seeds[@]}" ]]; then
    echo "TRNET_SEQ_LEN_GPU_IDS and TRNET_SEQ_LEN_SEEDS must have the same count" >&2
    exit 2
  fi
  local status=0
  local seq_len idx
  for seq_len in "${seq_lens[@]}"; do
    echo "[$(date -Is)] WAVE seq_len=$seq_len"
    local -a pids=()
    for idx in "${!gpus[@]}"; do
      local gpu="${gpus[$idx]//[[:space:]]/}"
      local seed="${seeds[$idx]}"
      local run_id
      run_id="$(run_id_for "$seq_len" "$seed")"
      train_one "$gpu" "$seq_len" "$seed" > "$LOG_DIR/${run_id}.log" 2>&1 &
      pids+=("$!")
      echo "$!" > "$LOG_DIR/${run_id}.pid"
    done
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        status=1
      fi
    done
    summarize || true
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
    setsid "$0" run "$SUFFIX" > "$LOG_DIR/run.log" 2>&1 < /dev/null &
    echo "$!" > "$PID_FILE"
    echo "Started TRNet seq_len ablation 4090 experiment."
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
