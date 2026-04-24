#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ACTION="${1:-start}"
INPUT_RUN_TAG="${2:-${RUN_TAG:-}}"

GPU_IDS="${AUDIO_SMOKE_GPU_IDS:-3,4}"
CONDA_ENV="${AUDIO_SMOKE_CONDA_ENV:-agri-mbt}"

TRAIN_CSV="${AUDIO_SMOKE_TRAIN_CSV:-data/b_deep_part_multimodal_full_clean_20260417/train.csv}"
VAL_CSV="${AUDIO_SMOKE_VAL_CSV:-data/b_deep_part_multimodal_full_clean_20260417/val.csv}"
TEST_CSV="${AUDIO_SMOKE_TEST_CSV:-data/b_deep_part_multimodal_full_clean_20260417/test.csv}"

EPOCHS="${AUDIO_SMOKE_EPOCHS:-1}"
BATCH_SIZE="${AUDIO_SMOKE_BATCH_SIZE:-4}"
NUM_WORKERS="${AUDIO_SMOKE_NUM_WORKERS:-0}"
SEQ_LEN="${AUDIO_SMOKE_SEQ_LEN:-64}"
STRIDE="${AUDIO_SMOKE_STRIDE:-10000}"
EVAL_STRIDE="${AUDIO_SMOKE_EVAL_STRIDE:-10000}"
MAX_TRAIN_BATCHES="${AUDIO_SMOKE_MAX_TRAIN_BATCHES:-1}"
MAX_EVAL_BATCHES="${AUDIO_SMOKE_MAX_EVAL_BATCHES:-1}"
MAX_TIME_GAP="${AUDIO_SMOKE_MAX_TIME_GAP:-1}"
LR="${AUDIO_SMOKE_LR:-0.0003}"
WEIGHT_DECAY="${AUDIO_SMOKE_WEIGHT_DECAY:-0.0001}"
CLASS_WEIGHT_POWER="${AUDIO_SMOKE_CLASS_WEIGHT_POWER:-0.5}"
LOSS_TYPE="${AUDIO_SMOKE_LOSS_TYPE:-weighted_ce}"
SKIP_CHECK="${AUDIO_SMOKE_SKIP_CHECK:-0}"

latest_run_tag() {
  local latest_dir
  latest_dir="$(ls -dt logs/audio_only_smoke_3090_* 2>/dev/null | head -n 1 || true)"
  if [[ -z "$latest_dir" ]]; then
    return 1
  fi
  basename "$latest_dir" | sed 's/^audio_only_smoke_3090_//'
}

if [[ -z "$INPUT_RUN_TAG" ]]; then
  case "$ACTION" in
    status|stop)
      RUN_TAG="$(latest_run_tag || true)"
      ;;
    *)
      RUN_TAG="$(date +%Y%m%d_%H%M%S)"
      ;;
  esac
else
  RUN_TAG="$INPUT_RUN_TAG"
fi

if [[ -z "${RUN_TAG:-}" ]]; then
  echo "No run found. Start one first: scripts/run_audio_only_3090_smoke.sh start" >&2
  exit 1
fi

SAVE_DIR="${AUDIO_SMOKE_SAVE_DIR:-experiments/audio_only_smoke_3090_${RUN_TAG}}"
LOG_DIR="${AUDIO_SMOKE_LOG_DIR:-logs/audio_only_smoke_3090_${RUN_TAG}}"
PID_FILE="$LOG_DIR/pid"
CMD_FILE="$LOG_DIR/cmd.sh"
RUN_LOG="$LOG_DIR/run.log"

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_audio_only_3090_smoke.sh start [RUN_TAG]
  scripts/run_audio_only_3090_smoke.sh run [RUN_TAG]
  scripts/run_audio_only_3090_smoke.sh status [RUN_TAG]
  scripts/run_audio_only_3090_smoke.sh stop [RUN_TAG]
  scripts/run_audio_only_3090_smoke.sh check [RUN_TAG]

Defaults:
  GPUs: 3,4 (physical indices, forced by CUDA_DEVICE_ORDER=PCI_BUS_ID)
  CSVs: data/b_deep_part_multimodal_full_clean_20260417/{train,val,test}.csv
  Background log: logs/audio_only_smoke_3090_<RUN_TAG>/run.log
  Outputs: experiments/audio_only_smoke_3090_<RUN_TAG>

Common env overrides:
  AUDIO_SMOKE_GPU_IDS, AUDIO_SMOKE_{TRAIN,VAL,TEST}_CSV, AUDIO_SMOKE_SAVE_DIR
  AUDIO_SMOKE_{EPOCHS,BATCH_SIZE,NUM_WORKERS,SEQ_LEN,STRIDE,EVAL_STRIDE}
  AUDIO_SMOKE_{MAX_TRAIN_BATCHES,MAX_EVAL_BATCHES}
USAGE
}

check_3090_gpus() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required" >&2
    exit 1
  fi
  local gpu_table id name
  gpu_table="$(
    env -u LD_PRELOAD -u PROXYCHAINS_CONF_FILE \
      nvidia-smi --query-gpu=index,name --format=csv,noheader,nounits
  )"
  IFS=',' read -r -a ids <<<"$GPU_IDS"
  for id in "${ids[@]}"; do
    id="${id//[[:space:]]/}"
    name="$(awk -F',' -v id="$id" '$1 + 0 == id {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}' <<<"$gpu_table")"
    if [[ -z "$name" ]]; then
      echo "GPU $id not found" >&2
      exit 1
    fi
    if [[ "$name" != *"3090"* ]]; then
      echo "Refusing to run: GPU $id is '$name', expected RTX 3090" >&2
      exit 1
    fi
  done
}

check_inputs() {
  check_3090_gpus
  for path in "$TRAIN_CSV" "$VAL_CSV" "$TEST_CSV"; do
    if [[ ! -f "$path" ]]; then
      echo "Missing CSV: $path" >&2
      exit 1
    fi
    echo "Found: $path"
  done
  echo "GPUs: $GPU_IDS"
  echo "Save dir: $SAVE_DIR"
  echo "Log dir: $LOG_DIR"
}

run_foreground() {
  mkdir -p "$LOG_DIR"
  if [[ "$SKIP_CHECK" == "1" ]]; then
    echo "Skipping input/GPU precheck because AUDIO_SMOKE_SKIP_CHECK=1"
  else
    check_inputs
  fi
  cat >"$CMD_FILE" <<EOF
env -u LD_PRELOAD -u PROXYCHAINS_CONF_FILE \\
  CUDA_DEVICE_ORDER=PCI_BUS_ID \\
  CUDA_VISIBLE_DEVICES=$GPU_IDS \\
  HF_HUB_OFFLINE=1 \\
  TRANSFORMERS_OFFLINE=1 \\
  HF_DATASETS_OFFLINE=1 \\
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
  /private/miniforge3/bin/conda run --no-capture-output -n $CONDA_ENV \\
  python -u src/train_ablation.py \\
    --mode audio_only \\
    --train-csv $TRAIN_CSV \\
    --val-csv $VAL_CSV \\
    --test-csv $TEST_CSV \\
    --save-dir $SAVE_DIR \\
    --seq-len $SEQ_LEN \\
    --stride $STRIDE \\
    --eval-stride $EVAL_STRIDE \\
    --context-mode causal \\
    --sampling-strategy fixed \\
    --epochs $EPOCHS \\
    --batch-size $BATCH_SIZE \\
    --num-workers $NUM_WORKERS \\
    --device cuda \\
    --all-gpus \\
    --lr $LR \\
    --weight-decay $WEIGHT_DECAY \\
    --class-weight-power $CLASS_WEIGHT_POWER \\
    --loss-type $LOSS_TYPE \\
    --max-train-batches $MAX_TRAIN_BATCHES \\
    --max-eval-batches $MAX_EVAL_BATCHES \\
    --max-time-gap $MAX_TIME_GAP
EOF
  chmod +x "$CMD_FILE"

  echo "[$(date -Is)] RUN_TAG=$RUN_TAG"
  echo "[$(date -Is)] Starting audio-only smoke on GPUs $GPU_IDS"
  echo "[$(date -Is)] Command file: $CMD_FILE"
  bash "$CMD_FILE"
}

start_background() {
  mkdir -p "$LOG_DIR"
  nohup "$0" run "$RUN_TAG" >"$RUN_LOG" 2>&1 &
  echo "$!" >"$PID_FILE"
  echo "Started in background."
  echo "PID: $(cat "$PID_FILE")"
  echo "Log: $RUN_LOG"
  echo "Save dir: $SAVE_DIR"
}

status_run() {
  if [[ -f "$PID_FILE" ]] && ps -p "$(cat "$PID_FILE")" >/dev/null 2>&1; then
    echo "running pid=$(cat "$PID_FILE")"
    echo "log=$RUN_LOG"
  elif [[ -f "$PID_FILE" ]]; then
    echo "not running pid=$(cat "$PID_FILE")"
    echo "log=$RUN_LOG"
  else
    echo "no pid file: $PID_FILE"
  fi
}

stop_run() {
  if [[ ! -f "$PID_FILE" ]]; then
    echo "No pid file: $PID_FILE"
    return 0
  fi
  local pid
  pid="$(cat "$PID_FILE")"
  if ps -p "$pid" >/dev/null 2>&1; then
    kill -TERM "$pid"
    echo "Sent TERM to pid=$pid"
  else
    echo "Process already stopped: pid=$pid"
  fi
}

case "$ACTION" in
  start)
    start_background
    ;;
  run)
    run_foreground
    ;;
  status)
    status_run
    ;;
  stop)
    stop_run
    ;;
  check)
    check_inputs
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
