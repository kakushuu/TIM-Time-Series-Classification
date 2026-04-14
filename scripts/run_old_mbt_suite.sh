#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_IDS="${GPU_IDS:-1,2,5,6}"
CSV_IN="${CSV_IN:-data/b_ocr_dataset/train/aligned_data.csv}"
CSV_27="${CSV_27:-data/old_mbt_20241018/aligned_data_27features.csv}"
OUT_DIR="${OUT_DIR:-experiments/old_mbt_20241018_suite}"
CONDA_BIN="${CONDA_BIN:-/private/miniforge3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-agri-mbt}"
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_SIZE="${TEST_SIZE:-0.2}"
LR="${LR:-3e-4}"
LOSS_TYPE="${LOSS_TYPE:-weighted_ce}"
TRAJ_ARCH="${TRAJ_ARCH:-bilstm}"
BILSTM_HIDDEN="${BILSTM_HIDDEN:-384}"
BILSTM_LAYERS="${BILSTM_LAYERS:-2}"

mkdir -p "$(dirname "$CSV_27")" "$OUT_DIR"

echo "================================================================================"
echo "Preparing old MBT 27-feature CSV"
echo "  input:  $CSV_IN"
echo "  output: $CSV_27"
echo "================================================================================"
"$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
  python scripts/compute_27_features.py \
    --input "$CSV_IN" \
    --output "$CSV_27"

run_mode() {
  local mode="$1"
  echo "================================================================================"
  echo "Running old MBT mode: $mode"
  echo "  physical GPUs: $GPU_IDS"
  echo "  output: $OUT_DIR"
  echo "================================================================================"
  (
    cd Multimodal-Fusion-with-Attention-Bottlenecks-main/MBT
    CUDA_VISIBLE_DEVICES="$GPU_IDS" "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
      python train_test.py \
        --gpu_id cuda:0 \
        --all_gpus \
        --mode "$mode" \
        --traj_arch "$TRAJ_ARCH" \
        --bilstm_hidden "$BILSTM_HIDDEN" \
        --bilstm_layers "$BILSTM_LAYERS" \
        --loss_type "$LOSS_TYPE" \
        --csv_file "../../$CSV_27" \
        --data_dir "../../" \
        --output_dir "../../$OUT_DIR" \
        --num_epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --test_size "$TEST_SIZE" \
        --lr "$LR"
  )
}

run_mode trajectory_only
run_mode image_only
run_mode multimodal

"$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
  python scripts/analyze_old_mbt_suite.py \
    --results-dir "$OUT_DIR" \
    --output-dir "$OUT_DIR/analysis"

echo "================================================================================"
echo "Old MBT suite finished"
echo "Analysis: $OUT_DIR/analysis"
echo "================================================================================"
