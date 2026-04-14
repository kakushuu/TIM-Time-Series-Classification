#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GPU_IDS="${AGRI_IMAGE_GPU_IDS:-1,2,5,6}"
CONDA_BIN="${AGRI_IMAGE_CONDA_BIN:-/private/miniforge3/bin/conda}"
CONDA_ENV="${AGRI_IMAGE_CONDA_ENV:-agri-mbt}"
RUN_ID="${AGRI_IMAGE_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SUITE_DIR="${AGRI_IMAGE_SUITE_DIR:-experiments/agri_image_autoresearch}"
SAVE_DIR="$SUITE_DIR/$RUN_ID"

TRAIN_CSV="${AGRI_IMAGE_TRAIN_CSV:-data/taif_20241018_split/train.csv}"
VAL_CSV="${AGRI_IMAGE_VAL_CSV:-data/taif_20241018_split/val.csv}"
TEST_CSV="${AGRI_IMAGE_TEST_CSV:-data/taif_20241018_split/test.csv}"
DURATION_STATS="${AGRI_IMAGE_DURATION_STATS:-experiments/new_adaptive_mbt_20241018_full/behavior_duration_analysis/duration_sampling_config.json}"

EPOCHS="${AGRI_IMAGE_EPOCHS:-30}"
BATCH_SIZE="${AGRI_IMAGE_BATCH_SIZE:-8}"
NUM_WORKERS="${AGRI_IMAGE_NUM_WORKERS:-8}"
SEQ_LEN="${AGRI_IMAGE_SEQ_LEN:-512}"
STRIDE="${AGRI_IMAGE_STRIDE:-20}"
EVAL_STRIDE="${AGRI_IMAGE_EVAL_STRIDE:-1}"
IMAGE_WINDOW_SIZE="${AGRI_IMAGE_WINDOW_SIZE:-9}"
IMAGE_SAMPLING="${AGRI_IMAGE_SAMPLING:-center}"
IMAGE_RADIUS="${AGRI_IMAGE_RADIUS:-8}"
IMAGE_RADIUS_MODE="${AGRI_IMAGE_RADIUS_MODE:-fixed}"
IMAGE_RADIUS_DURATION_SCALE="${AGRI_IMAGE_RADIUS_DURATION_SCALE:-0.5}"
IMAGE_RADIUS_CLASSES="${AGRI_IMAGE_RADIUS_CLASSES:-}"
IMAGE_TEMPORAL_POOL="${AGRI_IMAGE_TEMPORAL_POOL:-mean}"
IMAGE_TEMPORAL_DELTA="${AGRI_IMAGE_TEMPORAL_DELTA:-none}"
LR="${AGRI_IMAGE_LR:-0.0003}"
WEIGHT_DECAY="${AGRI_IMAGE_WEIGHT_DECAY:-0.0001}"
CLASS_WEIGHT_POWER="${AGRI_IMAGE_CLASS_WEIGHT_POWER:-0.5}"
TRAIN_SAMPLER="${AGRI_IMAGE_TRAIN_SAMPLER:-shuffle}"
SAMPLER_WEIGHT_POWER="${AGRI_IMAGE_SAMPLER_WEIGHT_POWER:-0.5}"
AUX_TARGET_CLASSES="${AGRI_IMAGE_AUX_TARGET_CLASSES:-}"
AUX_LOSS_WEIGHT="${AGRI_IMAGE_AUX_LOSS_WEIGHT:-0}"
AUX_POS_WEIGHT_POWER="${AGRI_IMAGE_AUX_POS_WEIGHT_POWER:-0.5}"
MAX_TRAIN_BATCHES="${AGRI_IMAGE_MAX_TRAIN_BATCHES:-0}"
MAX_EVAL_BATCHES="${AGRI_IMAGE_MAX_EVAL_BATCHES:-0}"
EARLY_STOP_VAL_MACRO_F1="${AGRI_IMAGE_EARLY_STOP_VAL_MACRO_F1:-0}"
EVAL_CHECKPOINT="${AGRI_IMAGE_EVAL_CHECKPOINT:-}"

mkdir -p "$SAVE_DIR"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_IDS" "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
  python src/train_ablation.py \
    --mode image_only \
    --train-csv "$TRAIN_CSV" \
    --val-csv "$VAL_CSV" \
    --test-csv "$TEST_CSV" \
    --save-dir "$SAVE_DIR" \
    --seq-len "$SEQ_LEN" \
    --stride "$STRIDE" \
    --eval-stride "$EVAL_STRIDE" \
    --context-mode causal \
    --sampling-strategy adaptive \
    --duration-stats "$DURATION_STATS" \
    --image-window-size "$IMAGE_WINDOW_SIZE" \
    --image-sampling "$IMAGE_SAMPLING" \
    --image-radius "$IMAGE_RADIUS" \
    --image-radius-mode "$IMAGE_RADIUS_MODE" \
    --image-radius-duration-scale "$IMAGE_RADIUS_DURATION_SCALE" \
    --image-radius-classes "$IMAGE_RADIUS_CLASSES" \
    --image-temporal-pool "$IMAGE_TEMPORAL_POOL" \
    --image-temporal-delta "$IMAGE_TEMPORAL_DELTA" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --class-weight-power "$CLASS_WEIGHT_POWER" \
    --train-sampler "$TRAIN_SAMPLER" \
    --sampler-weight-power "$SAMPLER_WEIGHT_POWER" \
    --aux-target-classes "$AUX_TARGET_CLASSES" \
    --aux-loss-weight "$AUX_LOSS_WEIGHT" \
    --aux-pos-weight-power "$AUX_POS_WEIGHT_POWER" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --device cuda \
    --all-gpus \
    --pretrained \
    --max-train-batches "$MAX_TRAIN_BATCHES" \
    --max-eval-batches "$MAX_EVAL_BATCHES" \
    --early-stop-val-macro-f1 "$EARLY_STOP_VAL_MACRO_F1" \
    ${EVAL_CHECKPOINT:+--eval-checkpoint "$EVAL_CHECKPOINT"}

python .agents/skills/agri-image-autoresearch/scripts/summarize_image_run.py \
  --summary "$SAVE_DIR/summary.json"
