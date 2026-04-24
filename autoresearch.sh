#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GPU_IDS="${AGRI_IMAGE_GPU_IDS:-1,2,5,6}"
CONDA_BIN="${AGRI_IMAGE_CONDA_BIN:-/private/miniforge3/bin/conda}"
CONDA_ENV="${AGRI_IMAGE_CONDA_ENV:-agri-mbt}"
RUN_ID="${AGRI_IMAGE_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SUITE_DIR="${AGRI_IMAGE_SUITE_DIR:-experiments/agri_image_autoresearch}"
SAVE_DIR="$SUITE_DIR/$RUN_ID"
MODE="${AGRI_IMAGE_MODE:-image_only}"

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
FEATURE_MODE="${AGRI_IMAGE_FEATURE_MODE:-engineered}"
TRAJ_ENCODER="${AGRI_IMAGE_TRAJ_ENCODER:-lstm}"
TRAJ_FEATURE_MAP_SIZE="${AGRI_IMAGE_TRAJ_FEATURE_MAP_SIZE:-6}"
PRETRAINED="${AGRI_IMAGE_PRETRAINED:-true}"
VISUAL_PRETRAINED_PATH="${AGRI_IMAGE_VISUAL_PRETRAINED_PATH:-}"
MAX_TIME_GAP="${AGRI_IMAGE_MAX_TIME_GAP:-1.0}"
LR="${AGRI_IMAGE_LR:-0.0003}"
WEIGHT_DECAY="${AGRI_IMAGE_WEIGHT_DECAY:-0.0001}"
CLASS_WEIGHT_POWER="${AGRI_IMAGE_CLASS_WEIGHT_POWER:-0.5}"
LOSS_TYPE="${AGRI_IMAGE_LOSS_TYPE:-weighted_ce}"
FOCAL_GAMMA="${AGRI_IMAGE_FOCAL_GAMMA:-2.0}"
CB_BETA="${AGRI_IMAGE_CB_BETA:-0.999}"
LABEL_SMOOTHING="${AGRI_IMAGE_LABEL_SMOOTHING:-0.0}"
TRAIN_SAMPLER="${AGRI_IMAGE_TRAIN_SAMPLER:-shuffle}"
SAMPLER_WEIGHT_POWER="${AGRI_IMAGE_SAMPLER_WEIGHT_POWER:-0.5}"
SAMPLER_BOOST_CLASSES="${AGRI_IMAGE_SAMPLER_BOOST_CLASSES:-}"
SAMPLER_BOOST_FACTOR="${AGRI_IMAGE_SAMPLER_BOOST_FACTOR:-1.0}"
AUX_TARGET_CLASSES="${AGRI_IMAGE_AUX_TARGET_CLASSES:-}"
AUX_LOSS_WEIGHT="${AGRI_IMAGE_AUX_LOSS_WEIGHT:-0}"
AUX_POS_WEIGHT_POWER="${AGRI_IMAGE_AUX_POS_WEIGHT_POWER:-0.5}"
MAX_TRAIN_BATCHES="${AGRI_IMAGE_MAX_TRAIN_BATCHES:-0}"
MAX_EVAL_BATCHES="${AGRI_IMAGE_MAX_EVAL_BATCHES:-0}"
EARLY_STOP_VAL_MACRO_F1="${AGRI_IMAGE_EARLY_STOP_VAL_MACRO_F1:-0}"
EARLY_STOP_PATIENCE="${AGRI_IMAGE_EARLY_STOP_PATIENCE:-0}"
EARLY_STOP_MIN_DELTA="${AGRI_IMAGE_EARLY_STOP_MIN_DELTA:-0.0}"
TEMPORAL_SMOOTHING="${AGRI_IMAGE_TEMPORAL_SMOOTHING:-none}"
SMOOTH_CLASSES="${AGRI_IMAGE_SMOOTH_CLASSES:-}"
SMOOTH_MIN_DURATION="${AGRI_IMAGE_SMOOTH_MIN_DURATION:-5}"
EVAL_CHECKPOINT="${AGRI_IMAGE_EVAL_CHECKPOINT:-}"

mkdir -p "$SAVE_DIR"

PRETRAINED_ARG="--pretrained"
case "${PRETRAINED,,}" in
  0|false|no|off)
    PRETRAINED_ARG="--no-pretrained"
    ;;
esac

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_IDS" "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
  python src/train_ablation.py \
    --mode "$MODE" \
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
    --feature-mode "$FEATURE_MODE" \
    --traj-encoder "$TRAJ_ENCODER" \
    --traj-feature-map-size "$TRAJ_FEATURE_MAP_SIZE" \
    --visual-pretrained-path "$VISUAL_PRETRAINED_PATH" \
    --max-time-gap "$MAX_TIME_GAP" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --class-weight-power "$CLASS_WEIGHT_POWER" \
    --loss-type "$LOSS_TYPE" \
    --focal-gamma "$FOCAL_GAMMA" \
    --cb-beta "$CB_BETA" \
    --label-smoothing "$LABEL_SMOOTHING" \
    --train-sampler "$TRAIN_SAMPLER" \
    --sampler-weight-power "$SAMPLER_WEIGHT_POWER" \
    --sampler-boost-classes "$SAMPLER_BOOST_CLASSES" \
    --sampler-boost-factor "$SAMPLER_BOOST_FACTOR" \
    --aux-target-classes "$AUX_TARGET_CLASSES" \
    --aux-loss-weight "$AUX_LOSS_WEIGHT" \
    --aux-pos-weight-power "$AUX_POS_WEIGHT_POWER" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --device cuda \
    --all-gpus \
    "$PRETRAINED_ARG" \
    --max-train-batches "$MAX_TRAIN_BATCHES" \
    --max-eval-batches "$MAX_EVAL_BATCHES" \
    --early-stop-val-macro-f1 "$EARLY_STOP_VAL_MACRO_F1" \
    --early-stop-patience "$EARLY_STOP_PATIENCE" \
    --early-stop-min-delta "$EARLY_STOP_MIN_DELTA" \
    --temporal-smoothing "$TEMPORAL_SMOOTHING" \
    --smooth-classes "$SMOOTH_CLASSES" \
    --smooth-min-duration "$SMOOTH_MIN_DURATION" \
    ${EVAL_CHECKPOINT:+--eval-checkpoint "$EVAL_CHECKPOINT"}

python .agents/skills/agri-image-autoresearch/scripts/summarize_image_run.py \
  --summary "$SAVE_DIR/summary.json"
