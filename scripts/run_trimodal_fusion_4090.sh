#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$ROOT_DIR/scripts/run_trimodal_fusion_4090.sh"
cd "$ROOT_DIR"

ACTION="${1:-start}"
RUN_TAG="${2:-${FUSION_RUN_TAG:-fusion_class_gate_20260420}}"
SUITE_DIR="${FUSION_SUITE_DIR:-experiments/trimodal_fusion_4090/$RUN_TAG}"
LOG_DIR="${FUSION_LOG_DIR:-logs/trimodal_fusion_4090/$RUN_TAG}"
PID_FILE="$LOG_DIR/pid"
ENV_FILE="$LOG_DIR/env.sh"
RUN_LOG="$LOG_DIR/run.log"

CONDA_BIN="${FUSION_CONDA_BIN:-/private/miniforge3/bin/conda}"
CONDA_ENV="${FUSION_CONDA_ENV:-agri-mbt}"
DATA_DIR="${FUSION_DATA_DIR:-data/b_deep_part_multimodal_full_clean_20260417}"
TRAIN_CSV="${FUSION_TRAIN_CSV:-$DATA_DIR/train.csv}"
VAL_CSV="${FUSION_VAL_CSV:-$DATA_DIR/val.csv}"
TEST_CSV="${FUSION_TEST_CSV:-$DATA_DIR/test.csv}"
DURATION_STATS="${FUSION_DURATION_STATS:-experiments/b_deep_part_duration_analysis/duration_sampling_config.json}"
REF_SUITE="${FUSION_REF_SUITE:-experiments/paper_4090_final_seed44}"
INIT_TRAJ_CHECKPOINT="${FUSION_INIT_TRAJ_CHECKPOINT:-$REF_SUITE/trnet/seed44/best.pt}"
INIT_IMAGE_CHECKPOINT="${FUSION_INIT_IMAGE_CHECKPOINT:-$REF_SUITE/image_best/seed44/best.pt}"
INIT_AUDIO_CHECKPOINT="${FUSION_INIT_AUDIO_CHECKPOINT:-$REF_SUITE/ast/seed44/best.pt}"
VISUAL_PRETRAINED_PATH="${FUSION_VISUAL_PRETRAINED_PATH:-/private/research/Agri-MBT/weights/vit_base_patch16_224.augreg2_in21k_ft_in1k/model.safetensors}"
AST_MODEL_NAME="${FUSION_AST_MODEL_NAME:-MIT/ast-finetuned-audioset-10-10-0.4593}"

GPU_IDS="${FUSION_GPU_IDS:-1,2,5}"
SEED="${FUSION_SEED:-44}"
SEQ_LEN="${FUSION_SEQ_LEN:-128}"
EPOCHS="${FUSION_EPOCHS:-24}"
PATIENCE="${FUSION_PATIENCE:-8}"
MIN_DELTA="${FUSION_MIN_DELTA:-0.001}"
BATCH_PER_GPU="${FUSION_BATCH_PER_GPU:-8}"
NUM_WORKERS="${FUSION_NUM_WORKERS:-4}"
LR="${FUSION_LR:-0.0001}"
WEIGHT_DECAY="${FUSION_WEIGHT_DECAY:-0.0001}"
FREEZE_ENCODERS_EPOCHS="${FUSION_FREEZE_ENCODERS_EPOCHS:-1}"
MAX_TRAIN_BATCHES="${FUSION_MAX_TRAIN_BATCHES:-0}"
MAX_EVAL_BATCHES="${FUSION_MAX_EVAL_BATCHES:-0}"
FORCE="${FUSION_FORCE:-0}"
EXPERIMENTS="${FUSION_EXPERIMENTS:-trimodal_class_gate}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_trimodal_fusion_4090.sh smoke [RUN_TAG]
  scripts/run_trimodal_fusion_4090.sh start [RUN_TAG]
  scripts/run_trimodal_fusion_4090.sh run RUN_TAG
  scripts/run_trimodal_fusion_4090.sh status [RUN_TAG]
  scripts/run_trimodal_fusion_4090.sh tail [RUN_TAG]
  scripts/run_trimodal_fusion_4090.sh stop [RUN_TAG]
  scripts/run_trimodal_fusion_4090.sh summarize [RUN_TAG]

Default experiment:
  trimodal_class_gate: class-adaptive logit-level trajectory/image/audio fusion.

Important overrides:
  FUSION_GPU_IDS=1,2,5
  FUSION_EXPERIMENTS='trimodal_class_gate'
  FUSION_INIT_IMAGE_CHECKPOINT=experiments/paper_4090_final_seed44/image_best/seed44/best.pt
  FUSION_FORCE=1
USAGE
}

gpu_count() {
  local selected="$1"
  IFS=',' read -r -a ids <<<"$selected"
  echo "${#ids[@]}"
}

total_batch() {
  echo "$(( BATCH_PER_GPU * $(gpu_count "$GPU_IDS") ))"
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

check_inputs() {
  validate_gpus
  for path in \
    "$TRAIN_CSV" "$VAL_CSV" "$TEST_CSV" "$DURATION_STATS" \
    "$INIT_TRAJ_CHECKPOINT" "$INIT_IMAGE_CHECKPOINT" "$INIT_AUDIO_CHECKPOINT" \
    "$VISUAL_PRETRAINED_PATH"; do
    if [[ ! -f "$path" ]]; then
      echo "Missing required input: $path" >&2
      exit 1
    fi
  done
}

common_args() {
  local save_dir="$1"
  ARGS=(
    --mode trimodal
    --train-csv "$TRAIN_CSV"
    --val-csv "$VAL_CSV"
    --test-csv "$TEST_CSV"
    --save-dir "$save_dir"
    --seq-len "$SEQ_LEN"
    --stride 20
    --eval-stride 1
    --context-mode causal
    --sampling-strategy adaptive
    --duration-stats "$DURATION_STATS"
    --adaptive-max-window "$SEQ_LEN"
    --feature-mode engineered
    --max-time-gap 1
    --epochs "$EPOCHS"
    --batch-size "$(total_batch)"
    --num-workers "$NUM_WORKERS"
    --device cuda
    --all-gpus
    --seed "$SEED"
    --lr "$LR"
    --weight-decay "$WEIGHT_DECAY"
    --class-weight-power 0.5
    --loss-type weighted_ce
    --early-stop-patience "$PATIENCE"
    --early-stop-min-delta "$MIN_DELTA"
    --max-train-batches "$MAX_TRAIN_BATCHES"
    --max-eval-batches "$MAX_EVAL_BATCHES"
    --traj-encoder trnet_seq
    --traj-feature-map-size 6
    --image-window-size 9
    --image-sampling nearest_causal
    --image-radius 8
    --image-temporal-pool gru
    --image-temporal-delta diff
    --visual-pretrained-path "$VISUAL_PRETRAINED_PATH"
    --ast-model-name "$AST_MODEL_NAME"
    --freeze-encoders-epochs "$FREEZE_ENCODERS_EPOCHS"
    --init-traj-checkpoint "$INIT_TRAJ_CHECKPOINT"
    --init-image-checkpoint "$INIT_IMAGE_CHECKPOINT"
    --init-audio-checkpoint "$INIT_AUDIO_CHECKPOINT"
    --no-skip-part-diagnostics
  )
}

variant_args() {
  local exp_id="$1"
  case "$exp_id" in
    trimodal_class_gate)
      ARGS+=(--fusion class_gate)
      ;;
    *)
      echo "Unknown trimodal fusion experiment: $exp_id" >&2
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
  common_args "$save_dir"
  variant_args "$exp_id"
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
  if [[ ! -f "$save_dir/summary.json" ]]; then
    echo "[$(date -Is)] FAIL exp=$exp_id seed=$SEED missing summary.json; see $log_file" >&2
    return 1
  fi
  echo "[$(date -Is)] DONE exp=$exp_id seed=$SEED"
}

summarize() {
  "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
    python scripts/summarize_paper_4090_suite.py \
      --suite-dir "$SUITE_DIR" \
      --output-dir "$LOG_DIR/results"
}

run_suite() {
  check_inputs
  mkdir -p "$LOG_DIR" "$SUITE_DIR"
  echo "[$(date -Is)] trimodal fusion suite"
  echo "run_tag=$RUN_TAG"
  echo "suite_dir=$SUITE_DIR"
  echo "log_dir=$LOG_DIR"
  echo "gpus=$GPU_IDS batch=$(total_batch) per_gpu=$BATCH_PER_GPU"
  echo "reference checkpoints: traj=$INIT_TRAJ_CHECKPOINT image=$INIT_IMAGE_CHECKPOINT audio=$INIT_AUDIO_CHECKPOINT"
  local status=0 exp_id
  for exp_id in $EXPERIMENTS; do
    if ! run_one "$exp_id"; then
      status=1
      echo "[$(date -Is)] FAIL exp=$exp_id"
    fi
    summarize || true
  done
  summarize || true
  exit "$status"
}

smoke() {
  local old_max_train="$MAX_TRAIN_BATCHES"
  local old_max_eval="$MAX_EVAL_BATCHES"
  local old_epochs="$EPOCHS"
  local old_batch="$BATCH_PER_GPU"
  MAX_TRAIN_BATCHES="${FUSION_SMOKE_MAX_TRAIN_BATCHES:-2}"
  MAX_EVAL_BATCHES="${FUSION_SMOKE_MAX_EVAL_BATCHES:-2}"
  EPOCHS=1
  BATCH_PER_GPU=1
  FORCE=1
  EXPERIMENTS="${FUSION_SMOKE_EXPERIMENTS:-trimodal_class_gate}"
  SUITE_DIR="$SUITE_DIR/smoke"
  LOG_DIR="$LOG_DIR/smoke"
  RUN_LOG="$LOG_DIR/run.log"
  run_suite
  MAX_TRAIN_BATCHES="$old_max_train"
  MAX_EVAL_BATCHES="$old_max_eval"
  EPOCHS="$old_epochs"
  BATCH_PER_GPU="$old_batch"
}

write_env_file() {
  mkdir -p "$LOG_DIR"
  {
    printf 'export FUSION_RUN_TAG=%q\n' "$RUN_TAG"
    printf 'export FUSION_SUITE_DIR=%q\n' "$SUITE_DIR"
    printf 'export FUSION_LOG_DIR=%q\n' "$LOG_DIR"
    printf 'export FUSION_CONDA_BIN=%q\n' "$CONDA_BIN"
    printf 'export FUSION_CONDA_ENV=%q\n' "$CONDA_ENV"
    printf 'export FUSION_DATA_DIR=%q\n' "$DATA_DIR"
    printf 'export FUSION_DURATION_STATS=%q\n' "$DURATION_STATS"
    printf 'export FUSION_REF_SUITE=%q\n' "$REF_SUITE"
    printf 'export FUSION_INIT_TRAJ_CHECKPOINT=%q\n' "$INIT_TRAJ_CHECKPOINT"
    printf 'export FUSION_INIT_IMAGE_CHECKPOINT=%q\n' "$INIT_IMAGE_CHECKPOINT"
    printf 'export FUSION_INIT_AUDIO_CHECKPOINT=%q\n' "$INIT_AUDIO_CHECKPOINT"
    printf 'export FUSION_VISUAL_PRETRAINED_PATH=%q\n' "$VISUAL_PRETRAINED_PATH"
    printf 'export FUSION_AST_MODEL_NAME=%q\n' "$AST_MODEL_NAME"
    printf 'export FUSION_GPU_IDS=%q\n' "$GPU_IDS"
    printf 'export FUSION_SEED=%q\n' "$SEED"
    printf 'export FUSION_SEQ_LEN=%q\n' "$SEQ_LEN"
    printf 'export FUSION_EPOCHS=%q\n' "$EPOCHS"
    printf 'export FUSION_PATIENCE=%q\n' "$PATIENCE"
    printf 'export FUSION_MIN_DELTA=%q\n' "$MIN_DELTA"
    printf 'export FUSION_BATCH_PER_GPU=%q\n' "$BATCH_PER_GPU"
    printf 'export FUSION_NUM_WORKERS=%q\n' "$NUM_WORKERS"
    printf 'export FUSION_LR=%q\n' "$LR"
    printf 'export FUSION_WEIGHT_DECAY=%q\n' "$WEIGHT_DECAY"
    printf 'export FUSION_FREEZE_ENCODERS_EPOCHS=%q\n' "$FREEZE_ENCODERS_EPOCHS"
    printf 'export FUSION_MAX_TRAIN_BATCHES=%q\n' "$MAX_TRAIN_BATCHES"
    printf 'export FUSION_MAX_EVAL_BATCHES=%q\n' "$MAX_EVAL_BATCHES"
    printf 'export FUSION_FORCE=%q\n' "$FORCE"
    printf 'export FUSION_EXPERIMENTS=%q\n' "$EXPERIMENTS"
  } > "$ENV_FILE"
}

status_run() {
  if [[ -f "$PID_FILE" ]] && ps -p "$(cat "$PID_FILE")" >/dev/null 2>&1; then
    echo "launcher running pid=$(cat "$PID_FILE")"
  elif [[ -f "$PID_FILE" ]]; then
    echo "launcher not running pid=$(cat "$PID_FILE")"
  else
    echo "no pid file: $PID_FILE"
  fi
  pgrep -af "run_trimodal_fusion_4090.sh run $RUN_TAG|$SUITE_DIR|src/train_ablation.py" || true
  echo "suite: $SUITE_DIR"
  echo "log: $RUN_LOG"
}

stop_run() {
  if [[ -f "$PID_FILE" ]] && ps -p "$(cat "$PID_FILE")" >/dev/null 2>&1; then
    kill -TERM "$(cat "$PID_FILE")"
    echo "sent TERM to launcher pid=$(cat "$PID_FILE")"
  else
    echo "no running launcher pid file"
  fi
}

case "$ACTION" in
  smoke)
    smoke
    ;;
  start)
    mkdir -p "$LOG_DIR"
    write_env_file
    nohup setsid env FUSION_CONFIG_FILE="$ENV_FILE" "$SCRIPT_PATH" run "$RUN_TAG" > "$RUN_LOG" 2>&1 < /dev/null &
    echo "$!" > "$PID_FILE"
    echo "Started trimodal fusion suite."
    echo "PID: $(cat "$PID_FILE")"
    echo "Log: $RUN_LOG"
    echo "Suite: $SUITE_DIR"
    ;;
  run)
    if [[ -n "${FUSION_CONFIG_FILE:-}" && -f "$FUSION_CONFIG_FILE" ]]; then
      # shellcheck disable=SC1090
      source "$FUSION_CONFIG_FILE"
      PID_FILE="$LOG_DIR/pid"
      ENV_FILE="$LOG_DIR/env.sh"
      RUN_LOG="$LOG_DIR/run.log"
    fi
    run_suite
    ;;
  status)
    status_run
    ;;
  tail)
    tail -f "$RUN_LOG"
    ;;
  stop)
    stop_run
    ;;
  summarize)
    summarize
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "unknown action: $ACTION" >&2
    usage
    exit 2
    ;;
esac
