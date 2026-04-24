#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ACTION="${1:-start}"
SUFFIX="${2:-${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}}"
GPU_IDS="${AGRI_IMAGE_GPU_IDS:-0,1,2,5}"
SPLIT_DIR="${AGRI_IMAGE_SPLIT_DIR:-data/b_deep_part_multimodal_full_clean_20260417}"
TRAIN_CSV="${AGRI_IMAGE_TRAIN_CSV:-$SPLIT_DIR/train.csv}"
VAL_CSV="${AGRI_IMAGE_VAL_CSV:-$SPLIT_DIR/val.csv}"
TEST_CSV="${AGRI_IMAGE_TEST_CSV:-$SPLIT_DIR/test.csv}"
DURATION_DIR="${AGRI_IMAGE_DURATION_DIR:-experiments/b_deep_part_duration_analysis}"
DURATION_STATS="${AGRI_IMAGE_DURATION_STATS:-$DURATION_DIR/duration_sampling_config.json}"
SUITE_DIR="${AGRI_IMAGE_SUITE_DIR:-experiments/agri_image_b_deep_part}"
LOG_DIR="${AGRI_IMAGE_LOG_DIR:-logs/agri_image_b_deep_part_${SUFFIX}}"
MANIFEST="$LOG_DIR/manifest.tsv"
PID_FILE="$LOG_DIR/pid"
SELECTED_EXPS="${SELECTED_EXPS:-exp-005-baseline exp-005-patience exp-005-smooth89 exp-005-boost46 exp-005-cbfocal exp-005-multimodal}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_b_deep_part_agri_exps.sh prepare [RUN_SUFFIX]
  scripts/run_b_deep_part_agri_exps.sh start [RUN_SUFFIX]
  scripts/run_b_deep_part_agri_exps.sh run RUN_SUFFIX
  scripts/run_b_deep_part_agri_exps.sh check [RUN_SUFFIX]
  scripts/run_b_deep_part_agri_exps.sh status [RUN_SUFFIX]
  scripts/run_b_deep_part_agri_exps.sh summarize [RUN_SUFFIX]

Environment overrides:
  AGRI_IMAGE_GPU_IDS      Default: 0,1,2,5
  AGRI_IMAGE_SPLIT_DIR    Default: data/b_deep_part_multimodal_full_clean_20260417
  SELECTED_EXPS           Default: exp-005-baseline exp-005-patience exp-005-smooth89 exp-005-boost46 exp-005-cbfocal exp-005-multimodal
                         Also supports: exp-006-trnet-seq
USAGE
}

run_id_for() {
  local exp_id="$1"
  printf '%s-%s' "$exp_id" "$SUFFIX"
}

description_for() {
  case "$1" in
    exp-005-baseline) echo "Exp-005 baseline: image_only GRU + diff, 10 epochs" ;;
    exp-005-patience) echo "GRU + diff with true patience early stopping" ;;
    exp-005-smooth89) echo "GRU + diff plus eval min-duration smoothing for classes 8/9" ;;
    exp-005-boost46) echo "GRU + diff plus class 4/6 boosted sampling" ;;
    exp-005-cbfocal) echo "GRU + diff plus class-balanced focal loss and label smoothing" ;;
    exp-005-multimodal) echo "Multimodal trajectory + image GRU + diff" ;;
    exp-006-trnet-seq) echo "Trajectory-only TRNet sequence: per-point 6x6 feature-map CNN + BiLSTM attention" ;;
    *) echo "custom experiment" ;;
  esac
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

prepare_data() {
  /private/miniforge3/bin/conda run --no-capture-output -n agri-mbt \
    python scripts/prepare_b_deep_part_splits.py \
      --output-dir "$SPLIT_DIR"
  /private/miniforge3/bin/conda run --no-capture-output -n agri-mbt \
    python scripts/analyze_behavior_durations.py \
      --input-csv "$TRAIN_CSV" \
      --output-dir "$DURATION_DIR"
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

plot_parts_if_present() {
  local run_id="$1"
  local pred="$SUITE_DIR/$run_id/predictions.csv"
  if [[ -f "$pred" ]]; then
    /private/miniforge3/bin/conda run --no-capture-output -n agri-mbt \
      python scripts/plot_b_deep_part_prediction_maps.py \
        --predictions "$pred" \
        --date 2024-10-27 \
        --output-dir "$SUITE_DIR/$run_id/part_diagnostics" || true
  fi
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
    --image-temporal-pool gru
    --image-temporal-delta diff
    --pretrained
    --max-time-gap 1
  )

  case "$exp_id" in
    exp-005-baseline)
      cmd+=(--mode image_only --epochs 10)
      ;;
    exp-005-patience)
      cmd+=(--mode image_only --epochs 12 --early-stop-patience 4 --early-stop-min-delta 0.001)
      ;;
    exp-005-smooth89)
      cmd+=(--mode image_only --epochs 12 --early-stop-patience 4 --early-stop-min-delta 0.001 --temporal-smoothing min_duration --smooth-classes 8,9 --smooth-min-duration 8)
      ;;
    exp-005-boost46)
      cmd+=(--mode image_only --epochs 12 --early-stop-patience 4 --early-stop-min-delta 0.001 --train-sampler class_boost --sampler-weight-power 0.25 --sampler-boost-classes 4,6 --sampler-boost-factor 3.0)
      ;;
    exp-005-cbfocal)
      cmd+=(--mode image_only --epochs 12 --early-stop-patience 4 --early-stop-min-delta 0.001 --loss-type cb_focal --focal-gamma 1.5 --cb-beta 0.999 --label-smoothing 0.05)
      ;;
    exp-005-multimodal)
      cmd+=(--mode multimodal --epochs 12 --early-stop-patience 4 --early-stop-min-delta 0.001)
      ;;
    exp-006-trnet-seq)
      cmd+=(--mode trajectory_only --epochs 12 --early-stop-patience 4 --early-stop-min-delta 0.001 --feature-mode engineered --traj-encoder trnet_seq --traj-feature-map-size 6)
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
  AGRI_IMAGE_SUITE_DIR="$SUITE_DIR" AGRI_IMAGE_DURATION_STATS="$DURATION_STATS" "${cmd[@]}"
  plot_parts_if_present "$run_id"
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

check_inputs() {
  require_4090_gpus
  for path in "$TRAIN_CSV" "$VAL_CSV" "$TEST_CSV" "$DURATION_STATS"; do
    if [[ ! -f "$path" ]]; then
      echo "Missing required input: $path" >&2
      exit 1
    fi
    echo "Found: $path"
  done
  echo "Selected experiments: $SELECTED_EXPS"
}

run_all() {
  check_inputs
  write_manifest
  echo "Run suffix: $SUFFIX"
  echo "GPU ids: $GPU_IDS"
  echo "Train CSV: $TRAIN_CSV"
  echo "Val CSV: $VAL_CSV"
  echo "Test CSV: $TEST_CSV"
  echo "Duration stats: $DURATION_STATS"
  echo "Suite dir: $SUITE_DIR"
  echo "Selected experiments: $SELECTED_EXPS"
  echo "Manifest: $MANIFEST"

  trap 'status=$?; echo "[$(date -Is)] run exited with status $status"; summarize_results || true; exit $status' EXIT
  for exp_id in $SELECTED_EXPS; do
    run_one "$exp_id"
  done
}

case "$ACTION" in
  prepare)
    prepare_data
    ;;
  start)
    mkdir -p "$LOG_DIR"
    nohup "$0" run "$SUFFIX" > "$LOG_DIR/run.log" 2>&1 &
    echo "$!" > "$PID_FILE"
    echo "Started B_deep_part experiments in background."
    echo "PID: $(cat "$PID_FILE")"
    echo "Log: $LOG_DIR/run.log"
    echo "Manifest: $MANIFEST"
    ;;
  run)
    run_all
    ;;
  check)
    check_inputs
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
