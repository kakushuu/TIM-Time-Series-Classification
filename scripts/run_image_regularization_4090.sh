#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ACTION="${1:-start}"
RUN_TAG="${2:-${IMAGE_REG_RUN_TAG:-image_reg_20260419}}"
SUITE_DIR="${IMAGE_REG_SUITE_DIR:-experiments/image_regularization_4090/$RUN_TAG}"
LOG_DIR="${IMAGE_REG_LOG_DIR:-logs/image_regularization_4090/$RUN_TAG}"
PID_FILE="$LOG_DIR/pid"
RUN_LOG="$LOG_DIR/run.log"

CONDA_BIN="${IMAGE_REG_CONDA_BIN:-/private/miniforge3/bin/conda}"
CONDA_ENV="${IMAGE_REG_CONDA_ENV:-agri-mbt}"
DATA_DIR="${IMAGE_REG_DATA_DIR:-data/b_deep_part_multimodal_full_clean_20260417}"
TRAIN_CSV="${IMAGE_REG_TRAIN_CSV:-$DATA_DIR/train.csv}"
VAL_CSV="${IMAGE_REG_VAL_CSV:-$DATA_DIR/val.csv}"
TEST_CSV="${IMAGE_REG_TEST_CSV:-$DATA_DIR/test.csv}"
VISUAL_PRETRAINED_PATH="${IMAGE_REG_VISUAL_PRETRAINED_PATH:-/private/research/Agri-MBT/weights/vit_base_patch16_224.augreg2_in21k_ft_in1k/model.safetensors}"
REG_DURATION_STATS="${IMAGE_REG_DURATION_STATS:-experiments/image_regularization_4090/duration_sampling_config_image_reg.json}"

GPU_IDS="${IMAGE_REG_GPU_IDS:-0,1,2,5}"
SEED="${IMAGE_REG_SEED:-44}"
SEQ_LEN="${IMAGE_REG_SEQ_LEN:-128}"
EPOCHS="${IMAGE_REG_EPOCHS:-16}"
PATIENCE="${IMAGE_REG_PATIENCE:-4}"
MIN_DELTA="${IMAGE_REG_MIN_DELTA:-0.001}"
IMAGE_BATCH_PER_GPU="${IMAGE_REG_BATCH_PER_GPU:-12}"
NUM_WORKERS="${IMAGE_REG_NUM_WORKERS:-16}"
JPEG_DRAFT_SIZE="${IMAGE_REG_JPEG_DRAFT_SIZE:-384}"
BATCH_TIMING="${IMAGE_REG_BATCH_TIMING:-0}"
MAX_TRAIN_BATCHES="${IMAGE_REG_MAX_TRAIN_BATCHES:-0}"
MAX_EVAL_BATCHES="${IMAGE_REG_MAX_EVAL_BATCHES:-0}"
FORCE="${IMAGE_REG_FORCE:-0}"
EXPERIMENTS="${IMAGE_REG_EXPERIMENTS:-freeze_visual_aug lowvlr_aug lowvlr_strong_aug sampling_cbfocal}"

OVERFIT_PATIENCE="${IMAGE_REG_OVERFIT_PATIENCE:-2}"
OVERFIT_MIN_EPOCH="${IMAGE_REG_OVERFIT_MIN_EPOCH:-6}"
OVERFIT_GAP="${IMAGE_REG_OVERFIT_GAP:-0.35}"
OVERFIT_VAL_LOSS_RISE="${IMAGE_REG_OVERFIT_VAL_LOSS_RISE:-0.45}"

gpu_count() {
  local selected="$1"
  IFS=',' read -r -a ids <<<"$selected"
  echo "${#ids[@]}"
}

total_batch() {
  echo "$(( IMAGE_BATCH_PER_GPU * $(gpu_count "$GPU_IDS") ))"
}

validate_gpus() {
  local table id name
  table="$(env -u LD_PRELOAD -u PROXYCHAINS_CONF_FILE nvidia-smi --query-gpu=index,name --format=csv,noheader,nounits)"
  IFS=',' read -r -a ids <<<"$GPU_IDS"
  for id in "${ids[@]}"; do
    id="${id//[[:space:]]/}"
    name="$(awk -F',' -v id="$id" '$1 + 0 == id {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}' <<<"$table")"
    if [[ "$name" != *"4090"* ]]; then
      echo "Selected GPU $id is '$name', not RTX 4090" >&2
      exit 1
    fi
  done
}

common_args() {
  local exp_id="$1"
  local save_dir="$2"
  ARGS=(
    --mode image_only
    --train-csv "$TRAIN_CSV"
    --val-csv "$VAL_CSV"
    --test-csv "$TEST_CSV"
    --save-dir "$save_dir"
    --seq-len "$SEQ_LEN"
    --stride 20
    --eval-stride 1
    --context-mode causal
    --sampling-strategy adaptive
    --duration-stats "$REG_DURATION_STATS"
    --adaptive-max-window "$SEQ_LEN"
    --feature-mode engineered
    --max-time-gap 1
    --epochs "$EPOCHS"
    --batch-size "$(total_batch)"
    --num-workers "$NUM_WORKERS"
    --device cuda
    --all-gpus
    --gpu-ids "$GPU_IDS"
    --seed "$SEED"
    --lr 0.0003
    --class-weight-power 0.5
    --loss-type weighted_ce
    --early-stop-patience "$PATIENCE"
    --early-stop-min-delta "$MIN_DELTA"
    --max-train-batches "$MAX_TRAIN_BATCHES"
    --max-eval-batches "$MAX_EVAL_BATCHES"
    --overfit-stop-patience "$OVERFIT_PATIENCE"
    --overfit-stop-min-epoch "$OVERFIT_MIN_EPOCH"
    --overfit-stop-gap "$OVERFIT_GAP"
    --overfit-stop-val-loss-rise "$OVERFIT_VAL_LOSS_RISE"
    --image-window-size 9
    --image-sampling nearest_causal
    --image-radius 8
    --image-temporal-pool gru
    --image-temporal-delta diff
    --image-jpeg-draft-size "$JPEG_DRAFT_SIZE"
    --visual-pretrained-path "$VISUAL_PRETRAINED_PATH"
    --skip-part-diagnostics
  )
  if [[ "$BATCH_TIMING" == "1" ]]; then
    ARGS+=(--batch-timing)
  fi
}

variant_args() {
  local exp_id="$1"
  case "$exp_id" in
    freeze_visual_aug)
      ARGS+=(
        --freeze-image-visual
        --image-augmentation light
        --image-frame-dropout 0.15
        --dropout 0.50
        --weight-decay 0.05
        --label-smoothing 0.05
      )
      ;;
    lowvlr_aug)
      ARGS+=(
        --image-visual-lr 0.00003
        --image-augmentation light
        --image-frame-dropout 0.10
        --dropout 0.45
        --weight-decay 0.03
        --label-smoothing 0.05
      )
      ;;
    lowvlr_strong_aug)
      ARGS+=(
        --image-visual-lr 0.00001
        --image-augmentation strong
        --image-frame-dropout 0.20
        --dropout 0.50
        --weight-decay 0.05
        --label-smoothing 0.10
      )
      ;;
    sampling_cbfocal)
      ARGS+=(
        --image-visual-lr 0.00003
        --image-augmentation light
        --image-frame-dropout 0.10
        --dropout 0.45
        --weight-decay 0.03
        --label-smoothing 0.05
        --loss-type cb_focal
        --focal-gamma 1.5
        --cb-beta 0.9999
        --train-sampler class_balanced
        --sampler-weight-power 0.25
      )
      ;;
    *)
      echo "Unknown image regularization experiment: $exp_id" >&2
      exit 2
      ;;
  esac
}

run_one() {
  local exp_id="$1"
  local save_dir="$SUITE_DIR/$exp_id/seed$SEED"
  local log_file="$LOG_DIR/${exp_id}_seed${SEED}.log"
  local cmd_file="$LOG_DIR/commands/${exp_id}_seed${SEED}.sh"
  if [[ "$FORCE" != "1" && -f "$save_dir/summary.json" ]]; then
    echo "[$(date -Is)] SKIP exp=$exp_id summary exists: $save_dir/summary.json"
    return 0
  fi
  mkdir -p "$save_dir" "$LOG_DIR/commands"
  common_args "$exp_id" "$save_dir"
  variant_args "$exp_id"
  CMD=(
    env -u LD_PRELOAD -u PROXYCHAINS_CONF_FILE
    CUDA_DEVICE_ORDER=PCI_BUS_ID
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    HF_DATASETS_OFFLINE=1
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV"
    python -u src/train_ablation.py
    "${ARGS[@]}"
  )
  {
    printf '#!/usr/bin/env bash\nset -euo pipefail\ncd %q\nexec' "$ROOT_DIR"
    printf ' %q' "${CMD[@]}"
    printf '\n'
  } > "$cmd_file"
  chmod +x "$cmd_file"
  echo "[$(date -Is)] START exp=$exp_id seed=$SEED gpus=$GPU_IDS batch=$(total_batch) save_dir=$save_dir"
  echo "[$(date -Is)] log=$log_file cmd=$cmd_file"
  "${CMD[@]}" > "$log_file" 2>&1
  echo "[$(date -Is)] DONE exp=$exp_id seed=$SEED"
}

summarize() {
  python - "$SUITE_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

suite = Path(sys.argv[1])
rows = []
for summary_path in sorted(suite.glob("*/seed*/summary.json")):
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    hist = data.get("history") or []
    best_gap = None
    final_gap = None
    if hist:
        final = hist[-1]
        final_gap = final["train"]["macro_f1"] - final["val"]["macro_f1"]
        best_epoch = None
        best_val = data.get("best_val_macro_f1", -1)
        for item in hist:
            if abs(item["val"]["macro_f1"] - best_val) < 1e-12:
                best_epoch = item
                break
        if best_epoch:
            best_gap = best_epoch["train"]["macro_f1"] - best_epoch["val"]["macro_f1"]
    test = data["test"]
    rows.append({
        "exp_id": summary_path.parents[1].name,
        "seed": summary_path.parent.name.replace("seed", ""),
        "epochs_ran": len(hist),
        "best_val_macro_f1": data.get("best_val_macro_f1"),
        "test_acc": test["acc"],
        "test_macro_f1": test["macro_f1"],
        "test_weighted_f1": test["weighted_f1"],
        "best_gap": best_gap,
        "final_gap": final_gap,
    })
if not rows:
    print(f"No completed summaries under {suite}")
    raise SystemExit(0)
out_dir = suite / "summary"
out_dir.mkdir(parents=True, exist_ok=True)
out_csv = out_dir / "image_regularization_summary.csv"
with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(out_csv)
print("| exp_id | epochs | best val macro-F1 | test macro-F1 | test acc | weighted-F1 | best gap | final gap |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
def fmt(value):
    return "" if value is None else f"{value:.4f}"
for row in sorted(rows, key=lambda r: r["test_macro_f1"], reverse=True):
    print(
        f"| {row['exp_id']} | {row['epochs_ran']} | {row['best_val_macro_f1']:.4f} | "
        f"{row['test_macro_f1']:.4f} | {row['test_acc']:.4f} | {row['test_weighted_f1']:.4f} | "
        f"{fmt(row['best_gap'])} | {fmt(row['final_gap'])} |"
    )
PY
}

run_suite() {
  validate_gpus
  mkdir -p "$LOG_DIR"
  {
    echo "[$(date -Is)] image regularization suite"
    echo "run_tag=$RUN_TAG"
    echo "suite_dir=$SUITE_DIR"
    echo "log_dir=$LOG_DIR"
    echo "gpus=$GPU_IDS batch=$(total_batch) per_gpu=$IMAGE_BATCH_PER_GPU"
    echo "num_workers=$NUM_WORKERS jpeg_draft_size=$JPEG_DRAFT_SIZE"
    echo "batch_timing=$BATCH_TIMING"
    echo "max_train_batches=$MAX_TRAIN_BATCHES max_eval_batches=$MAX_EVAL_BATCHES"
    echo "experiments=$EXPERIMENTS"
    echo "overfit_guard=patience:$OVERFIT_PATIENCE min_epoch:$OVERFIT_MIN_EPOCH gap:$OVERFIT_GAP val_loss_rise:$OVERFIT_VAL_LOSS_RISE"
  } | tee -a "$RUN_LOG"
  for exp_id in $EXPERIMENTS; do
    run_one "$exp_id" 2>&1 | tee -a "$RUN_LOG"
  done
  summarize 2>&1 | tee -a "$RUN_LOG"
}

case "$ACTION" in
  run)
    run_suite
    ;;
  start)
    mkdir -p "$LOG_DIR"
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "Already running: pid=$(cat "$PID_FILE")"
      exit 0
    fi
    nohup "$0" run "$RUN_TAG" > "$RUN_LOG" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Started image regularization suite: pid=$(cat "$PID_FILE") log=$RUN_LOG"
    ;;
  status)
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "RUNNING pid=$(cat "$PID_FILE")"
    else
      echo "NOT RUNNING"
    fi
    [[ -f "$RUN_LOG" ]] && tail -n 30 "$RUN_LOG"
    ;;
  tail)
    tail -f "$RUN_LOG"
    ;;
  summarize)
    summarize
    ;;
  stop)
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      kill -TERM "$(cat "$PID_FILE")"
      echo "Stopped pid=$(cat "$PID_FILE")"
    else
      echo "No running suite"
    fi
    ;;
  *)
    echo "Usage: $0 {start|run|status|tail|summarize|stop} [RUN_TAG]" >&2
    exit 2
    ;;
esac
