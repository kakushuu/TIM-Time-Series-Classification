#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$ROOT_DIR/scripts/run_ast_audio_baseline_3090.sh"
cd "$ROOT_DIR"

ACTION="${1:-start}"
INPUT_RUN_TAG="${2:-${RUN_TAG:-}}"
RUN_PREFIX="ast_audio_baseline_3090"

latest_run_tag() {
  local latest_dir
  latest_dir="$(ls -dt "logs/${RUN_PREFIX}_"* 2>/dev/null | head -n 1 || true)"
  if [[ -z "$latest_dir" ]]; then
    return 1
  fi
  basename "$latest_dir" | sed "s/^${RUN_PREFIX}_//"
}

if [[ -z "$INPUT_RUN_TAG" ]]; then
  case "$ACTION" in
    status|stop|tail)
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
  echo "No AST audio baseline run found. Start one first." >&2
  exit 1
fi

GPU_IDS="${AST_AUDIO_GPU_IDS:-3,4}"
CONDA_ENV="${AST_AUDIO_CONDA_ENV:-agri-mbt}"
CONDA_BIN="${AST_AUDIO_CONDA_BIN:-/private/miniforge3/bin/conda}"

DATA_DIR="${AST_AUDIO_DATA_DIR:-data/b_deep_part_multimodal_full_clean_20260417}"
TRAIN_CSV="${AST_AUDIO_TRAIN_CSV:-$DATA_DIR/train.csv}"
VAL_CSV="${AST_AUDIO_VAL_CSV:-$DATA_DIR/val.csv}"
TEST_CSV="${AST_AUDIO_TEST_CSV:-$DATA_DIR/test.csv}"

SAVE_DIR="${AST_AUDIO_SAVE_DIR:-experiments/${RUN_PREFIX}_${RUN_TAG}}"
LOG_DIR="${AST_AUDIO_LOG_DIR:-logs/${RUN_PREFIX}_${RUN_TAG}}"
PID_FILE="$LOG_DIR/pid"
ENV_FILE="$LOG_DIR/env.sh"
CMD_FILE="$LOG_DIR/cmd.sh"
RUN_LOG="$LOG_DIR/run.log"

if [[ -n "${AST_AUDIO_CONFIG_FILE:-}" && -f "$AST_AUDIO_CONFIG_FILE" ]]; then
  # Background runs source the exact config captured by `start`.
  # shellcheck disable=SC1090
  source "$AST_AUDIO_CONFIG_FILE"
  PID_FILE="$LOG_DIR/pid"
  ENV_FILE="$LOG_DIR/env.sh"
  CMD_FILE="$LOG_DIR/cmd.sh"
  RUN_LOG="$LOG_DIR/run.log"
fi

EPOCHS="${AST_AUDIO_EPOCHS:-${EPOCHS:-12}}"
BATCH_SIZE="${AST_AUDIO_BATCH_SIZE:-${BATCH_SIZE:-4}}"
NUM_WORKERS="${AST_AUDIO_NUM_WORKERS:-${NUM_WORKERS:-4}}"
SEQ_LEN="${AST_AUDIO_SEQ_LEN:-${SEQ_LEN:-64}}"
STRIDE="${AST_AUDIO_STRIDE:-${STRIDE:-1}}"
EVAL_STRIDE="${AST_AUDIO_EVAL_STRIDE:-${EVAL_STRIDE:-1}}"
MAX_TRAIN_BATCHES="${AST_AUDIO_MAX_TRAIN_BATCHES:-${MAX_TRAIN_BATCHES:-0}}"
MAX_EVAL_BATCHES="${AST_AUDIO_MAX_EVAL_BATCHES:-${MAX_EVAL_BATCHES:-0}}"
MAX_TIME_GAP="${AST_AUDIO_MAX_TIME_GAP:-${MAX_TIME_GAP:-1}}"
LR="${AST_AUDIO_LR:-${LR:-0.0003}}"
WEIGHT_DECAY="${AST_AUDIO_WEIGHT_DECAY:-${WEIGHT_DECAY:-0.0001}}"
CLASS_WEIGHT_POWER="${AST_AUDIO_CLASS_WEIGHT_POWER:-${CLASS_WEIGHT_POWER:-0.5}}"
LOSS_TYPE="${AST_AUDIO_LOSS_TYPE:-${LOSS_TYPE:-weighted_ce}}"
AST_MODEL_NAME="${AST_AUDIO_AST_MODEL_NAME:-${AST_MODEL_NAME:-MIT/ast-finetuned-audioset-10-10-0.4593}}"
SKIP_CHECK="${AST_AUDIO_SKIP_CHECK:-${SKIP_CHECK:-0}}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_ast_audio_baseline_3090.sh start [RUN_TAG]
  scripts/run_ast_audio_baseline_3090.sh run [RUN_TAG]
  scripts/run_ast_audio_baseline_3090.sh status [RUN_TAG]
  scripts/run_ast_audio_baseline_3090.sh tail [RUN_TAG]
  scripts/run_ast_audio_baseline_3090.sh stop [RUN_TAG]
  scripts/run_ast_audio_baseline_3090.sh check [RUN_TAG]

Defaults:
  GPUs: 3,4 (physical indices with CUDA_DEVICE_ORDER=PCI_BUS_ID)
  Data: data/b_deep_part_multimodal_full_clean_20260417/{train,val,test}.csv
  Mode: audio_only, full data, no max-batch smoke limit
  Epochs: 12
  Output: experiments/ast_audio_baseline_3090_<RUN_TAG>
  Log: logs/ast_audio_baseline_3090_<RUN_TAG>/run.log

Common overrides:
  AST_AUDIO_GPU_IDS=3,4
  AST_AUDIO_EPOCHS=12
  AST_AUDIO_BATCH_SIZE=4
  AST_AUDIO_NUM_WORKERS=4
  AST_AUDIO_STRIDE=1
  AST_AUDIO_EVAL_STRIDE=1
  AST_AUDIO_DATA_DIR=data/b_deep_part_multimodal_full_clean_20260417
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
      echo "GPU $id not found by nvidia-smi" >&2
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
  echo "Run tag: $RUN_TAG"
  echo "GPUs: $GPU_IDS"
  echo "Save dir: $SAVE_DIR"
  echo "Log dir: $LOG_DIR"
  echo "Epochs: $EPOCHS"
  echo "Batch size: $BATCH_SIZE"
  echo "Stride/eval stride: $STRIDE/$EVAL_STRIDE"
  echo "Max train/eval batches: $MAX_TRAIN_BATCHES/$MAX_EVAL_BATCHES"
}

write_env_file() {
  mkdir -p "$LOG_DIR"
  {
    printf 'export RUN_TAG=%q\n' "$RUN_TAG"
    printf 'export GPU_IDS=%q\n' "$GPU_IDS"
    printf 'export CONDA_ENV=%q\n' "$CONDA_ENV"
    printf 'export CONDA_BIN=%q\n' "$CONDA_BIN"
    printf 'export TRAIN_CSV=%q\n' "$TRAIN_CSV"
    printf 'export VAL_CSV=%q\n' "$VAL_CSV"
    printf 'export TEST_CSV=%q\n' "$TEST_CSV"
    printf 'export SAVE_DIR=%q\n' "$SAVE_DIR"
    printf 'export LOG_DIR=%q\n' "$LOG_DIR"
    printf 'export EPOCHS=%q\n' "$EPOCHS"
    printf 'export BATCH_SIZE=%q\n' "$BATCH_SIZE"
    printf 'export NUM_WORKERS=%q\n' "$NUM_WORKERS"
    printf 'export SEQ_LEN=%q\n' "$SEQ_LEN"
    printf 'export STRIDE=%q\n' "$STRIDE"
    printf 'export EVAL_STRIDE=%q\n' "$EVAL_STRIDE"
    printf 'export MAX_TRAIN_BATCHES=%q\n' "$MAX_TRAIN_BATCHES"
    printf 'export MAX_EVAL_BATCHES=%q\n' "$MAX_EVAL_BATCHES"
    printf 'export MAX_TIME_GAP=%q\n' "$MAX_TIME_GAP"
    printf 'export LR=%q\n' "$LR"
    printf 'export WEIGHT_DECAY=%q\n' "$WEIGHT_DECAY"
    printf 'export CLASS_WEIGHT_POWER=%q\n' "$CLASS_WEIGHT_POWER"
    printf 'export LOSS_TYPE=%q\n' "$LOSS_TYPE"
    printf 'export AST_MODEL_NAME=%q\n' "$AST_MODEL_NAME"
    printf 'export SKIP_CHECK=%q\n' "$SKIP_CHECK"
  } > "$ENV_FILE"
}

build_command() {
  CMD=(
    env -u LD_PRELOAD -u PROXYCHAINS_CONF_FILE
    CUDA_DEVICE_ORDER=PCI_BUS_ID
    "CUDA_VISIBLE_DEVICES=$GPU_IDS"
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    HF_DATASETS_OFFLINE=1
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV"
    python -u src/train_ablation.py
    --mode audio_only
    --train-csv "$TRAIN_CSV"
    --val-csv "$VAL_CSV"
    --test-csv "$TEST_CSV"
    --save-dir "$SAVE_DIR"
    --seq-len "$SEQ_LEN"
    --stride "$STRIDE"
    --eval-stride "$EVAL_STRIDE"
    --context-mode causal
    --sampling-strategy fixed
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --device cuda
    --all-gpus
    --lr "$LR"
    --weight-decay "$WEIGHT_DECAY"
    --class-weight-power "$CLASS_WEIGHT_POWER"
    --loss-type "$LOSS_TYPE"
    --ast-model-name "$AST_MODEL_NAME"
    --max-train-batches "$MAX_TRAIN_BATCHES"
    --max-eval-batches "$MAX_EVAL_BATCHES"
    --max-time-gap "$MAX_TIME_GAP"
  )
}

write_cmd_file() {
  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd %q\n' "$ROOT_DIR"
    printf 'exec'
    printf ' %q' "${CMD[@]}"
    printf '\n'
  } > "$CMD_FILE"
  chmod +x "$CMD_FILE"
}

run_foreground() {
  trap 'status=$?; echo "[$(date -Is)] AST audio baseline exited with status $status"; exit $status' EXIT

  if [[ "$SKIP_CHECK" == "1" ]]; then
    echo "Skipping precheck because AST_AUDIO_SKIP_CHECK=1"
  else
    check_inputs
  fi

  mkdir -p "$LOG_DIR" "$SAVE_DIR"
  build_command
  write_cmd_file

  echo "[$(date -Is)] RUN_TAG=$RUN_TAG"
  echo "[$(date -Is)] Starting AST audio baseline on physical GPUs $GPU_IDS"
  echo "[$(date -Is)] Command file: $CMD_FILE"
  "${CMD[@]}"
}

start_background() {
  if [[ "$SKIP_CHECK" != "1" ]]; then
    check_inputs
  fi

  SKIP_CHECK=1
  write_env_file
  nohup setsid env AST_AUDIO_CONFIG_FILE="$ENV_FILE" "$SCRIPT_PATH" run "$RUN_TAG" > "$RUN_LOG" 2>&1 < /dev/null &
  echo "$!" > "$PID_FILE"

  echo "Started AST audio baseline in background."
  echo "PID: $(cat "$PID_FILE")"
  echo "Run tag: $RUN_TAG"
  echo "Log: $RUN_LOG"
  echo "Save dir: $SAVE_DIR"
}

status_run() {
  if [[ -f "$PID_FILE" ]] && ps -p "$(cat "$PID_FILE")" >/dev/null 2>&1; then
    echo "launcher running pid=$(cat "$PID_FILE")"
  elif [[ -f "$PID_FILE" ]]; then
    echo "launcher not running pid=$(cat "$PID_FILE")"
  else
    echo "no pid file: $PID_FILE"
  fi

  if pgrep -af -- "$SAVE_DIR" >/dev/null 2>&1; then
    echo "training process:"
    pgrep -af -- "$SAVE_DIR" || true
  fi

  echo "log=$RUN_LOG"
  echo "save_dir=$SAVE_DIR"
}

stop_run() {
  if pgrep -af -- "$SAVE_DIR" >/dev/null 2>&1; then
    pkill -TERM -f -- "$SAVE_DIR" || true
    echo "Sent TERM to training process matching save dir: $SAVE_DIR"
  fi

  if [[ -f "$PID_FILE" ]] && ps -p "$(cat "$PID_FILE")" >/dev/null 2>&1; then
    kill -TERM "$(cat "$PID_FILE")" || true
    echo "Sent TERM to launcher pid=$(cat "$PID_FILE")"
  fi
}

tail_log() {
  if [[ ! -f "$RUN_LOG" ]]; then
    echo "No log file yet: $RUN_LOG" >&2
    exit 1
  fi
  tail -f "$RUN_LOG"
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
  tail)
    tail_log
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
