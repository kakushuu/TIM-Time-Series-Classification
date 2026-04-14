#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Full rerun entrypoint for the new adaptive sampling + pretrained image + MBT-style fusion code.
# Override any variable from the shell, e.g.
#   GPU_IDS=1,2,5,6 SUITE_DIR=experiments/my_run ./scripts/run_new_adaptive_mbt_full.sh

RUN_NAME="${RUN_NAME:-new_adaptive_mbt_20241018_$(date +%Y%m%d_%H%M%S)}"
SUITE_DIR="${SUITE_DIR:-experiments/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$SUITE_DIR/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/full_run.log}"

GPU_IDS="${GPU_IDS:-1,2,5,6}"
CONDA_BIN="${CONDA_BIN:-/private/miniforge3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-agri-mbt}"

TRAIN_CSV="${TRAIN_CSV:-data/taif_20241018_split/train.csv}"
VAL_CSV="${VAL_CSV:-data/taif_20241018_split/val.csv}"
TEST_CSV="${TEST_CSV:-data/taif_20241018_split/test.csv}"

SEQ_LEN="${SEQ_LEN:-512}"
STRIDE="${STRIDE:-20}"
EVAL_STRIDE="${EVAL_STRIDE:-1}"
CONTEXT_MODE="${CONTEXT_MODE:-causal}"
SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-adaptive}"

IMAGE_WINDOW_SIZE="${IMAGE_WINDOW_SIZE:-5}"
IMAGE_SAMPLING="${IMAGE_SAMPLING:-center}"
IMAGE_RADIUS="${IMAGE_RADIUS:-4}"

PRETRAINED="${PRETRAINED:-1}"
MULTI_FUSION="${MULTI_FUSION:-mbt}"
FUSION_LAYERS="${FUSION_LAYERS:-2}"
FUSION_HEADS="${FUSION_HEADS:-8}"
NUM_LATENTS="${NUM_LATENTS:-4}"
TRAJ_LR="${TRAJ_LR:-3e-4}"
IMAGE_LR="${IMAGE_LR:-3e-4}"
MULTI_LR="${MULTI_LR:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
FREEZE_ENCODERS_EPOCHS="${FREEZE_ENCODERS_EPOCHS:-3}"
INIT_FROM_UNIMODAL="${INIT_FROM_UNIMODAL:-1}"

TRAJ_EPOCHS="${TRAJ_EPOCHS:-50}"
IMAGE_EPOCHS="${IMAGE_EPOCHS:-30}"
MULTI_EPOCHS="${MULTI_EPOCHS:-30}"

TRAJ_BATCH_SIZE="${TRAJ_BATCH_SIZE:-32}"
IMAGE_BATCH_SIZE="${IMAGE_BATCH_SIZE:-8}"
MULTI_BATCH_SIZE="${MULTI_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"

DURATION_DIR="${DURATION_DIR:-$SUITE_DIR/behavior_duration_analysis}"
DURATION_STATS="${DURATION_STATS:-$DURATION_DIR/duration_sampling_config.json}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================================"
echo "Full new adaptive MBT run"
echo "================================================================================"
echo "Start time:        $(date '+%Y-%m-%d %H:%M:%S')"
echo "Suite dir:         $SUITE_DIR"
echo "Log file:          $LOG_FILE"
echo "GPUs:              $GPU_IDS"
echo "Train CSV:         $TRAIN_CSV"
echo "Val CSV:           $VAL_CSV"
echo "Test CSV:          $TEST_CSV"
echo "Seq/context:       seq_len=$SEQ_LEN context=$CONTEXT_MODE train_stride=$STRIDE eval_stride=$EVAL_STRIDE"
echo "Sampling:          $SAMPLING_STRATEGY"
echo "Images:            pretrained=$PRETRAINED image_window_size=$IMAGE_WINDOW_SIZE image_radius=$IMAGE_RADIUS"
echo "Fusion:            multimodal=$MULTI_FUSION layers=$FUSION_LAYERS heads=$FUSION_HEADS latents=$NUM_LATENTS"
echo "LR/clip:           traj=$TRAJ_LR image=$IMAGE_LR multimodal=$MULTI_LR grad_clip=$GRAD_CLIP"
echo "Multimodal init:   init_from_unimodal=$INIT_FROM_UNIMODAL freeze_encoders_epochs=$FREEZE_ENCODERS_EPOCHS"
echo "Epochs:            traj=$TRAJ_EPOCHS image=$IMAGE_EPOCHS multimodal=$MULTI_EPOCHS"
echo "Batch size:        traj=$TRAJ_BATCH_SIZE image=$IMAGE_BATCH_SIZE multimodal=$MULTI_BATCH_SIZE"
echo "================================================================================"

for csv in "$TRAIN_CSV" "$VAL_CSV" "$TEST_CSV"; do
  if [[ ! -f "$csv" ]]; then
    echo "Missing CSV: $csv"
    echo "Please prepare splits first, for example:"
    echo "  $CONDA_BIN run --no-capture-output -n $CONDA_ENV python scripts/prepare_taif_splits.py"
    exit 1
  fi
done

echo "[1/4] Syntax check"
"$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
  python -m py_compile \
    src/train_ablation.py \
    scripts/analyze_behavior_durations.py \
    scripts/analyze_ablation_suite.py

echo "[2/4] Behavior duration analysis"
"$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
  python scripts/analyze_behavior_durations.py \
    --input-csv "$TRAIN_CSV" \
    --output-dir "$DURATION_DIR" \
    --max-window "$SEQ_LEN"

echo "[3/4] Train trajectory/image/multimodal ablation suite"
GPU_IDS="$GPU_IDS" \
SUITE_DIR="$SUITE_DIR" \
SEQ_LEN="$SEQ_LEN" \
STRIDE="$STRIDE" \
EVAL_STRIDE="$EVAL_STRIDE" \
CONTEXT_MODE="$CONTEXT_MODE" \
SAMPLING_STRATEGY="$SAMPLING_STRATEGY" \
DURATION_DIR="$DURATION_DIR" \
DURATION_STATS="$DURATION_STATS" \
IMAGE_WINDOW_SIZE="$IMAGE_WINDOW_SIZE" \
IMAGE_SAMPLING="$IMAGE_SAMPLING" \
IMAGE_RADIUS="$IMAGE_RADIUS" \
PRETRAINED="$PRETRAINED" \
MULTI_FUSION="$MULTI_FUSION" \
FUSION_LAYERS="$FUSION_LAYERS" \
FUSION_HEADS="$FUSION_HEADS" \
NUM_LATENTS="$NUM_LATENTS" \
TRAJ_LR="$TRAJ_LR" \
IMAGE_LR="$IMAGE_LR" \
MULTI_LR="$MULTI_LR" \
GRAD_CLIP="$GRAD_CLIP" \
FREEZE_ENCODERS_EPOCHS="$FREEZE_ENCODERS_EPOCHS" \
INIT_FROM_UNIMODAL="$INIT_FROM_UNIMODAL" \
TRAIN_CSV="$TRAIN_CSV" \
VAL_CSV="$VAL_CSV" \
TEST_CSV="$TEST_CSV" \
CONDA_BIN="$CONDA_BIN" \
CONDA_ENV="$CONDA_ENV" \
TRAJ_EPOCHS="$TRAJ_EPOCHS" \
IMAGE_EPOCHS="$IMAGE_EPOCHS" \
MULTI_EPOCHS="$MULTI_EPOCHS" \
TRAJ_BATCH_SIZE="$TRAJ_BATCH_SIZE" \
IMAGE_BATCH_SIZE="$IMAGE_BATCH_SIZE" \
MULTI_BATCH_SIZE="$MULTI_BATCH_SIZE" \
NUM_WORKERS="$NUM_WORKERS" \
./scripts/run_ablation_suite.sh

echo "[4/4] Artifact summary"
find "$SUITE_DIR" -maxdepth 3 -type f \( \
  -name 'summary.json' -o \
  -name 'metrics.csv' -o \
  -name 'predictions.csv' -o \
  -name 'training_curves.png' -o \
  -name 'confusion_matrix.png' -o \
  -name 'per_class_metrics.png' -o \
  -name 'spatial_errors.png' -o \
  -name 'analysis_report.md' -o \
  -name 'overall_metrics.png' -o \
  -name 'spatial_error_density.png' \
\) | sort

echo "================================================================================"
echo "Finished:          $(date '+%Y-%m-%d %H:%M:%S')"
echo "Suite dir:         $SUITE_DIR"
echo "Analysis dir:      $SUITE_DIR/analysis"
echo "Log file:          $LOG_FILE"
echo "================================================================================"
