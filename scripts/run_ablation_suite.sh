#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_IDS="${GPU_IDS:-1,2,5,6}"
SUITE_DIR="${SUITE_DIR:-experiments/ablation_20241018_suite}"
SEQ_LEN="${SEQ_LEN:-512}"
STRIDE="${STRIDE:-20}"
EVAL_STRIDE="${EVAL_STRIDE:-1}"
CONTEXT_MODE="${CONTEXT_MODE:-causal}"
SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-adaptive}"
DURATION_DIR="${DURATION_DIR:-$SUITE_DIR/behavior_duration_analysis}"
DURATION_STATS="${DURATION_STATS:-$DURATION_DIR/duration_sampling_config.json}"
IMAGE_WINDOW_SIZE="${IMAGE_WINDOW_SIZE:-9}"
IMAGE_SAMPLING="${IMAGE_SAMPLING:-center}"
IMAGE_RADIUS="${IMAGE_RADIUS:-8}"
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
TRAIN_CSV="${TRAIN_CSV:-data/taif_20241018_split/train.csv}"
VAL_CSV="${VAL_CSV:-data/taif_20241018_split/val.csv}"
TEST_CSV="${TEST_CSV:-data/taif_20241018_split/test.csv}"
CONDA_BIN="${CONDA_BIN:-/private/miniforge3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-agri-mbt}"

mkdir -p "$SUITE_DIR"

if [[ "$SAMPLING_STRATEGY" == "adaptive" && ! -f "$DURATION_STATS" ]]; then
  mkdir -p "$DURATION_DIR"
  "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
    python scripts/analyze_behavior_durations.py \
      --input-csv "$TRAIN_CSV" \
      --output-dir "$DURATION_DIR" \
      --max-window "$SEQ_LEN"
fi

run_exp() {
  local mode="$1"
  local epochs="$2"
  local batch_size="$3"
  local fusion="$4"
  local lr="$5"
  local save_dir="$SUITE_DIR/$mode"
  local pretrained_arg="--pretrained"
  local init_args=()
  if [[ "$PRETRAINED" == "0" ]]; then
    pretrained_arg="--no-pretrained"
  fi
  if [[ "$mode" == "multimodal" ]]; then
    init_args+=(--freeze-encoders-epochs "$FREEZE_ENCODERS_EPOCHS")
    if [[ "$INIT_FROM_UNIMODAL" == "1" ]]; then
      init_args+=(--init-traj-checkpoint "$SUITE_DIR/trajectory_only/best.pt")
      init_args+=(--init-image-checkpoint "$SUITE_DIR/image_only/best.pt")
    fi
  fi

  echo "================================================================================"
  echo "Running $mode"
  echo "  GPUs: $GPU_IDS"
  echo "  save_dir: $save_dir"
  echo "================================================================================"

  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
    python src/train_ablation.py \
      --mode "$mode" \
      --train-csv "$TRAIN_CSV" \
      --val-csv "$VAL_CSV" \
      --test-csv "$TEST_CSV" \
      --save-dir "$save_dir" \
      --seq-len "$SEQ_LEN" \
      --stride "$STRIDE" \
      --eval-stride "$EVAL_STRIDE" \
      --context-mode "$CONTEXT_MODE" \
      --sampling-strategy "$SAMPLING_STRATEGY" \
      --duration-stats "$DURATION_STATS" \
      --image-window-size "$IMAGE_WINDOW_SIZE" \
      --image-sampling "$IMAGE_SAMPLING" \
      --image-radius "$IMAGE_RADIUS" \
      --fusion "$fusion" \
      --fusion-layers "$FUSION_LAYERS" \
      --fusion-heads "$FUSION_HEADS" \
      --num-latents "$NUM_LATENTS" \
      --epochs "$epochs" \
      --lr "$lr" \
      --grad-clip "$GRAD_CLIP" \
      --batch-size "$batch_size" \
      --num-workers "${NUM_WORKERS:-8}" \
      --device cuda \
      --all-gpus \
      "$pretrained_arg" \
      "${init_args[@]}"
}

run_exp trajectory_only "${TRAJ_EPOCHS:-50}" "${TRAJ_BATCH_SIZE:-32}" "concat" "$TRAJ_LR"
run_exp image_only "${IMAGE_EPOCHS:-30}" "${IMAGE_BATCH_SIZE:-8}" "concat" "$IMAGE_LR"
run_exp multimodal "${MULTI_EPOCHS:-30}" "${MULTI_BATCH_SIZE:-4}" "$MULTI_FUSION" "$MULTI_LR"

"$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
  python scripts/analyze_ablation_suite.py \
    --suite-dir "$SUITE_DIR" \
    --output-dir "$SUITE_DIR/analysis"

echo "================================================================================"
echo "Ablation suite finished"
echo "Analysis: $SUITE_DIR/analysis"
echo "================================================================================"
