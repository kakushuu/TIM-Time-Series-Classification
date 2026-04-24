#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ACTION="${1:-start}"
SUFFIX="${2:-${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}}"
GPU_IDS="${AGRI_IMAGE_GPU_IDS:-0,1,2,5}"
SPLIT_DIR="${AGRI_IMAGE_SPLIT_DIR:-data/taif_20241018_20_train_20241022_eval}"
TRAIN_CSV="${AGRI_IMAGE_TRAIN_CSV:-$SPLIT_DIR/train.csv}"
VAL_CSV="${AGRI_IMAGE_VAL_CSV:-$SPLIT_DIR/val.csv}"
TEST_CSV="${AGRI_IMAGE_TEST_CSV:-$SPLIT_DIR/test.csv}"
SUITE_DIR="${AGRI_IMAGE_SUITE_DIR:-experiments/agri_image_autoresearch}"
LOG_DIR="${AGRI_IMAGE_LOG_DIR:-logs/agri_image_selected_${SUFFIX}}"
MANIFEST="$LOG_DIR/manifest.tsv"
PID_FILE="$LOG_DIR/pid"
SELECTED_EXPS="${SELECTED_EXPS:-exp-003 exp-002 exp-008 exp-005}"

if [[ "$ACTION" != "-h" && "$ACTION" != "--help" && "$ACTION" != "help" && $# -gt 2 ]]; then
  echo "Too many arguments: $*" >&2
  echo "Use a single suffix without spaces, for example: train18_20_eval22" >&2
  exit 2
fi

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_selected_agri_image_exps.sh start [RUN_SUFFIX]
  scripts/run_selected_agri_image_exps.sh run RUN_SUFFIX
  scripts/run_selected_agri_image_exps.sh check [RUN_SUFFIX]
  scripts/run_selected_agri_image_exps.sh status [RUN_SUFFIX]
  scripts/run_selected_agri_image_exps.sh summarize [RUN_SUFFIX]

Environment overrides:
  AGRI_IMAGE_GPU_IDS      Default: 0,1,2,5
  AGRI_IMAGE_SPLIT_DIR    Default: data/taif_20241018_20_train_20241022_eval
  SELECTED_EXPS           Default: exp-003 exp-002 exp-008 exp-005

The default split trains on 2024-10-18/19/20 and uses 2024-10-22 for both
validation and test. The default GPU list is the physical RTX 4090 set.
USAGE
}

require_4090_gpus() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required to verify GPU models" >&2
    exit 1
  fi

  local gpu_table id name
  gpu_table="$(nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits)"
  IFS=',' read -r -a ids <<<"$GPU_IDS"
  for id in "${ids[@]}"; do
    id="${id//[[:space:]]/}"
    name="$(awk -F',' -v id="$id" '$1 + 0 == id {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}' <<<"$gpu_table")"
    if [[ -z "$name" ]]; then
      echo "Refusing to run: GPU $id was not found by nvidia-smi" >&2
      exit 1
    fi
    if [[ "$name" != *"4090"* ]]; then
      echo "Refusing to run: GPU $id is '$name', not an RTX 4090" >&2
      exit 1
    fi
  done
}

write_manifest() {
  mkdir -p "$LOG_DIR"
  {
    printf 'exp_id\trun_id\tdescription\n'
    for exp_id in $SELECTED_EXPS; do
      printf '%s\t%s\t%s\n' "$exp_id" "$(run_id_for "$exp_id")" "$(description_for "$exp_id")"
    done
  } > "$MANIFEST"
}

run_id_for() {
  local exp_id="$1"
  printf '%s-1022eval-%s' "$exp_id" "$SUFFIX"
}

description_for() {
  case "$1" in
    exp-003) echo "GRU temporal pooling; previous best kept candidate by generalization balance" ;;
    exp-002) echo "Transformer temporal pooling; previous best validation macro F1" ;;
    exp-008) echo "GRU plus auxiliary weak-class target loss; previous best test macro F1" ;;
    exp-005) echo "GRU plus temporal embedding delta; previous better weak-class floor candidate" ;;
    exp-007) echo "GRU plus class-balanced sampler; optional comparison, previously overfit" ;;
    *) echo "custom experiment" ;;
  esac
}

run_one() {
  local exp_id="$1"
  local run_id
  run_id="$(run_id_for "$exp_id")"

  local -a cmd=(
    scripts/run_agri_image_gpu.sh
    --run-id "$run_id"
    --gpu-ids "$GPU_IDS"
    --train-csv "$TRAIN_CSV"
    --val-csv "$VAL_CSV"
    --test-csv "$TEST_CSV"
  )

  case "$exp_id" in
    exp-003)
      cmd+=(--image-temporal-pool gru --early-stop-val-macro-f1 0.50)
      ;;
    exp-002)
      cmd+=(--image-temporal-pool transformer)
      ;;
    exp-008)
      cmd+=(
        --epochs 6
        --image-temporal-pool gru
        --aux-target-classes 1,2,4,5,6,8,9
        --aux-loss-weight 0.25
        --aux-pos-weight-power 0.5
      )
      ;;
    exp-005)
      cmd+=(--epochs 10 --image-temporal-pool gru --image-temporal-delta diff)
      ;;
    exp-007)
      cmd+=(--epochs 10 --image-temporal-pool gru --train-sampler class_balanced --sampler-weight-power 0.5)
      ;;
    *)
      echo "Unknown experiment id: $exp_id" >&2
      exit 2
      ;;
  esac

  echo "[$(date -Is)] START $exp_id -> $run_id"
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
  echo "[$(date -Is)] DONE $exp_id -> $run_id"
}

summarize_results() {
  mkdir -p "$LOG_DIR/results"
  if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing manifest: $MANIFEST" >&2
    return 1
  fi
  python scripts/summarize_selected_agri_image_exps.py \
    --manifest "$MANIFEST" \
    --suite-dir "$SUITE_DIR" \
    --output-dir "$LOG_DIR/results"
}

active_run_ids() {
  if [[ -f "$MANIFEST" ]]; then
    awk -F'\t' 'NR > 1 {print $2}' "$MANIFEST"
  else
    for exp_id in $SELECTED_EXPS; do
      run_id_for "$exp_id"
    done
  fi
}

run_all() {
  require_4090_gpus
  for path in "$TRAIN_CSV" "$VAL_CSV" "$TEST_CSV"; do
    if [[ ! -f "$path" ]]; then
      echo "Missing CSV: $path" >&2
      exit 1
    fi
  done

  write_manifest
  echo "Run suffix: $SUFFIX"
  echo "GPU ids: $GPU_IDS"
  echo "Train CSV: $TRAIN_CSV"
  echo "Val CSV: $VAL_CSV"
  echo "Test CSV: $TEST_CSV"
  echo "Selected experiments: $SELECTED_EXPS"
  echo "Manifest: $MANIFEST"

  trap 'status=$?; echo "[$(date -Is)] run exited with status $status"; summarize_results || true; exit $status' EXIT

  for exp_id in $SELECTED_EXPS; do
    run_one "$exp_id"
  done
}

check_inputs() {
  require_4090_gpus
  echo "4090 GPU check passed for: $GPU_IDS"
  for path in "$TRAIN_CSV" "$VAL_CSV" "$TEST_CSV"; do
    if [[ ! -f "$path" ]]; then
      echo "Missing CSV: $path" >&2
      exit 1
    fi
    echo "Found CSV: $path"
  done
  echo "Selected experiments: $SELECTED_EXPS"
}

case "$ACTION" in
  start)
    mkdir -p "$LOG_DIR"
    nohup "$0" run "$SUFFIX" > "$LOG_DIR/run.log" 2>&1 &
    echo "$!" > "$PID_FILE"
    echo "Started selected experiments in background."
    echo "PID: $(cat "$PID_FILE")"
    echo "Log: $LOG_DIR/run.log"
    echo "Manifest: $MANIFEST"
    ;;
  run)
    run_all
    exit $?
    ;;
  check)
    check_inputs
    ;;
  status)
    any_running=0
    if [[ -f "$PID_FILE" ]]; then
      pid="$(cat "$PID_FILE")"
      if ps -p "$pid" >/dev/null 2>&1; then
        echo "running pid=$pid"
        any_running=1
      else
        echo "not running pid=$pid"
      fi
    else
      echo "No pid file: $PID_FILE"
    fi
    while IFS= read -r run_id; do
      matches="$(ps -ef | awk -v pat="$run_id" 'index($0, pat) && !index($0, "awk -v pat") {print}')"
      if [[ -n "$matches" ]]; then
        echo "active run process for $run_id:"
        echo "$matches"
        any_running=1
      fi
    done < <(active_run_ids)
    if [[ "$any_running" -eq 0 ]]; then
      echo "no active run processes found for suffix=$SUFFIX"
    fi
    [[ -f "$LOG_DIR/run.log" ]] && tail -80 "$LOG_DIR/run.log"
    ;;
  summarize)
    summarize_results
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
