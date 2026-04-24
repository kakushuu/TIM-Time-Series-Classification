#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$ROOT_DIR/scripts/run_paper_4090_suite.sh"
cd "$ROOT_DIR"

ACTION="${1:-start}"
RUN_TAG="${2:-${PAPER_RUN_TAG:-paper_20260417}}"
SUITE_DIR="${PAPER_SUITE_DIR:-experiments/paper_4090_${RUN_TAG#paper_}}"
LOG_DIR="${PAPER_LOG_DIR:-logs/paper_4090_${RUN_TAG#paper_}}"
PID_FILE="$LOG_DIR/pid"
ENV_FILE="$LOG_DIR/env.sh"
RUN_LOG="$LOG_DIR/run.log"

CONDA_BIN="${PAPER_CONDA_BIN:-/private/miniforge3/bin/conda}"
CONDA_ENV="${PAPER_CONDA_ENV:-agri-mbt}"
DATA_DIR="${PAPER_DATA_DIR:-data/b_deep_part_multimodal_full_clean_20260417}"
TRAIN_CSV="${PAPER_TRAIN_CSV:-$DATA_DIR/train.csv}"
VAL_CSV="${PAPER_VAL_CSV:-$DATA_DIR/val.csv}"
TEST_CSV="${PAPER_TEST_CSV:-$DATA_DIR/test.csv}"
DURATION_STATS="${PAPER_DURATION_STATS:-experiments/b_deep_part_duration_analysis/duration_sampling_config.json}"
VISUAL_PRETRAINED_PATH="${PAPER_VISUAL_PRETRAINED_PATH:-/private/research/Agri-MBT/weights/vit_base_patch16_224.augreg2_in21k_ft_in1k/model.safetensors}"
AST_MODEL_NAME="${PAPER_AST_MODEL_NAME:-MIT/ast-finetuned-audioset-10-10-0.4593}"

GPU_IDS_REQUEST="${PAPER_GPU_IDS:-auto}"
FALLBACK_GPU_IDS="${PAPER_FALLBACK_GPU_IDS:-1,2,5}"
TRUST_GPU_IDS_ON_QUERY_FAIL="${PAPER_TRUST_GPU_IDS_ON_QUERY_FAIL:-1}"
GPU_BUSY_MEMORY_MB="${PAPER_GPU_BUSY_MEMORY_MB:-1024}"
MIN_GPUS="${PAPER_MIN_GPUS:-2}"
SEEDS="${PAPER_SEEDS:-44}"
EXPERIMENTS="${PAPER_EXPERIMENTS-image_best trnet ast trimodal_concat}"
FORCE="${PAPER_FORCE:-0}"
SKIP_CHECK="${PAPER_SKIP_CHECK:-0}"

SEQ_LEN="${PAPER_SEQ_LEN:-128}"
NUM_WORKERS="${PAPER_NUM_WORKERS:-4}"
TRNET_EPOCHS="${PAPER_TRNET_EPOCHS:-20}"
IMAGE_EPOCHS="${PAPER_IMAGE_EPOCHS:-20}"
AST_EPOCHS="${PAPER_AST_EPOCHS:-20}"
TRIMODAL_EPOCHS="${PAPER_TRIMODAL_EPOCHS:-24}"
TRNET_PATIENCE="${PAPER_TRNET_PATIENCE:-6}"
IMAGE_PATIENCE="${PAPER_IMAGE_PATIENCE:-6}"
AST_PATIENCE="${PAPER_AST_PATIENCE:-6}"
TRIMODAL_PATIENCE="${PAPER_TRIMODAL_PATIENCE:-8}"
TRNET_MIN_DELTA="${PAPER_TRNET_MIN_DELTA:-0.001}"
IMAGE_MIN_DELTA="${PAPER_IMAGE_MIN_DELTA:-0.001}"
AST_MIN_DELTA="${PAPER_AST_MIN_DELTA:-0.001}"
TRIMODAL_MIN_DELTA="${PAPER_TRIMODAL_MIN_DELTA:-0.001}"
TRNET_BATCH_PER_GPU="${PAPER_TRNET_BATCH_PER_GPU:-16}"
IMAGE_BATCH_PER_GPU="${PAPER_IMAGE_BATCH_PER_GPU:-16}"
AST_BATCH_PER_GPU="${PAPER_AST_BATCH_PER_GPU:-16}"
TRIMODAL_BATCH_PER_GPU="${PAPER_TRIMODAL_BATCH_PER_GPU:-8}"
TRNET_LR="${PAPER_TRNET_LR:-0.0003}"
IMAGE_LR="${PAPER_IMAGE_LR:-0.0003}"
AST_LR="${PAPER_AST_LR:-0.0003}"
TRIMODAL_LR="${PAPER_TRIMODAL_LR:-0.0001}"

if [[ -n "${PAPER_CONFIG_FILE:-}" && -f "$PAPER_CONFIG_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$PAPER_CONFIG_FILE"
  PID_FILE="$LOG_DIR/pid"
  ENV_FILE="$LOG_DIR/env.sh"
  RUN_LOG="$LOG_DIR/run.log"
fi

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_paper_4090_suite.sh smoke [RUN_TAG]
  scripts/run_paper_4090_suite.sh start [RUN_TAG]
  scripts/run_paper_4090_suite.sh run RUN_TAG
  scripts/run_paper_4090_suite.sh status [RUN_TAG]
  scripts/run_paper_4090_suite.sh tail [RUN_TAG]
  scripts/run_paper_4090_suite.sh stop [RUN_TAG]
  scripts/run_paper_4090_suite.sh summarize [RUN_TAG]

Defaults:
  Data: data/b_deep_part_multimodal_full_clean_20260417/{train,val,test}.csv
  GPU selection: PAPER_GPU_IDS=auto, using idle RTX 4090 cards only.
  Current suite: experiments/paper_4090_<RUN_TAG without paper_ prefix>

Important overrides:
  PAPER_GPU_IDS=1,2,5              Use explicit physical 4090 IDs.
  PAPER_GPU_IDS=auto               Exclude 4090s with active compute processes.
  PAPER_FALLBACK_GPU_IDS=1,2,5      Used if nvidia-smi GPU table is temporarily unavailable.
  PAPER_TRNET_BATCH_PER_GPU=16     Total batch is per-GPU batch times selected GPU count.
  PAPER_IMAGE_BATCH_PER_GPU=16
  PAPER_AST_BATCH_PER_GPU=16
  PAPER_TRIMODAL_BATCH_PER_GPU=8
  PAPER_SEEDS=44                   Use the best previous trajectory seed by default.
  PAPER_EXPERIMENTS='image_best trnet ast trimodal_concat'
  PAPER_FORCE=1                    Re-run completed summary.json runs.
USAGE
}

query_gpu_table() {
  local gpu_table attempt
  gpu_table=""
  for attempt in 1 2 3 4 5; do
    if gpu_table="$(env -u LD_PRELOAD -u PROXYCHAINS_CONF_FILE nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader 2>&1)"; then
      printf '%s\n' "$gpu_table"
      return 0
    fi
    echo "nvidia-smi gpu table failed (attempt $attempt/5): $gpu_table" >&2
    sleep 2
  done
  echo "Unable to query NVIDIA GPU table" >&2
  return 1
}

busy_gpu_uuids() {
  env -u LD_PRELOAD -u PROXYCHAINS_CONF_FILE nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' -v limit="$GPU_BUSY_MEMORY_MB" '
      {
        uuid=$1
        mem=$2 + 0
        gsub(/^[ \t]+|[ \t]+$/, "", uuid)
        if (uuid != "" && mem >= limit) print uuid
      }' || true
}

select_gpu_ids() {
  local gpu_table busy
  if [[ "$GPU_IDS_REQUEST" != "auto" ]]; then
    printf '%s\n' "$GPU_IDS_REQUEST"
    return 0
  fi
  if ! gpu_table="$(query_gpu_table)"; then
    echo "Using PAPER_FALLBACK_GPU_IDS=$FALLBACK_GPU_IDS because GPU table query failed" >&2
    printf '%s\n' "$FALLBACK_GPU_IDS"
    return 0
  fi
  busy="$(busy_gpu_uuids)"
  awk -F',' -v busy="$busy" '
    BEGIN {
      split(busy, b, "\n")
      for (i in b) if (b[i] != "") busy_map[b[i]] = 1
    }
    {
      idx=$1
      uuid=$2
      name=$3
      gsub(/^[ \t]+|[ \t]+$/, "", idx)
      gsub(/^[ \t]+|[ \t]+$/, "", uuid)
      gsub(/^[ \t]+|[ \t]+$/, "", name)
      if (name ~ /4090/ && !(uuid in busy_map)) {
        if (out != "") out = out ","
        out = out idx
      }
    }
    END { print out }
  ' <<<"$gpu_table"
}

require_selected_4090_gpus() {
  local selected gpu_table count id name
  selected="$1"
  if [[ -z "$selected" ]]; then
    echo "No idle RTX 4090 GPUs selected. Set PAPER_GPU_IDS explicitly if needed." >&2
    exit 1
  fi
  if ! gpu_table="$(query_gpu_table)"; then
    if [[ "$TRUST_GPU_IDS_ON_QUERY_FAIL" == "1" ]]; then
      echo "Warning: GPU validation skipped because nvidia-smi query failed; trusting selected GPUs: $selected" >&2
      return 0
    fi
    return 1
  fi
  IFS=',' read -r -a ids <<<"$selected"
  count="${#ids[@]}"
  if (( count < MIN_GPUS )); then
    echo "Selected only $count RTX 4090 GPU(s), below PAPER_MIN_GPUS=$MIN_GPUS: $selected" >&2
    exit 1
  fi
  for id in "${ids[@]}"; do
    id="${id//[[:space:]]/}"
    name="$(awk -F',' -v id="$id" '$1 + 0 == id {gsub(/^[ \t]+|[ \t]+$/, "", $3); print $3}' <<<"$gpu_table")"
    if [[ "$name" != *"4090"* ]]; then
      echo "Selected GPU $id is '$name', not an RTX 4090" >&2
      exit 1
    fi
  done
}

gpu_count() {
  local selected="$1"
  IFS=',' read -r -a ids <<<"$selected"
  echo "${#ids[@]}"
}

total_batch() {
  local per_gpu="$1"
  local selected="$2"
  echo "$(( per_gpu * $(gpu_count "$selected") ))"
}

run_dir_for() {
  local exp_id="$1"
  local seed="$2"
  printf '%s/%s/seed%s' "$SUITE_DIR" "$exp_id" "$seed"
}

checkpoint_for() {
  local exp_id="$1"
  local seed="$2"
  printf '%s/best.pt' "$(run_dir_for "$exp_id" "$seed")"
}

base_args() {
  local save_dir="$1"
  local seed="$2"
  local selected_gpus="$3"
  local batch_size="$4"
  local epochs="$5"
  local lr="$6"
  local patience="$7"
  local min_delta="$8"
  ARGS=(
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
    --epochs "$epochs"
    --batch-size "$batch_size"
    --num-workers "$NUM_WORKERS"
    --device cuda
    --all-gpus
    --seed "$seed"
    --lr "$lr"
    --weight-decay 0.0001
    --class-weight-power 0.5
    --loss-type weighted_ce
    --early-stop-patience "$patience"
    --early-stop-min-delta "$min_delta"
  )
  VISIBLE_GPUS="$selected_gpus"
}

build_args_for() {
  local exp_id="$1"
  local seed="$2"
  local selected_gpus="$3"
  local save_dir="$4"
  local batch
  case "$exp_id" in
    trnet)
      batch="$(total_batch "$TRNET_BATCH_PER_GPU" "$selected_gpus")"
      base_args "$save_dir" "$seed" "$selected_gpus" "$batch" "$TRNET_EPOCHS" "$TRNET_LR" "$TRNET_PATIENCE" "$TRNET_MIN_DELTA"
      ARGS=(--mode trajectory_only "${ARGS[@]}" --traj-encoder trnet_seq --traj-feature-map-size 6)
      ;;
    image_best)
      batch="$(total_batch "$IMAGE_BATCH_PER_GPU" "$selected_gpus")"
      base_args "$save_dir" "$seed" "$selected_gpus" "$batch" "$IMAGE_EPOCHS" "$IMAGE_LR" "$IMAGE_PATIENCE" "$IMAGE_MIN_DELTA"
      ARGS=(
        --mode image_only
        "${ARGS[@]}"
        --image-window-size 9
        --image-sampling nearest_causal
        --image-radius 8
        --image-temporal-pool gru
        --image-temporal-delta diff
        --visual-pretrained-path "$VISUAL_PRETRAINED_PATH"
      )
      ;;
    ast)
      batch="$(total_batch "$AST_BATCH_PER_GPU" "$selected_gpus")"
      base_args "$save_dir" "$seed" "$selected_gpus" "$batch" "$AST_EPOCHS" "$AST_LR" "$AST_PATIENCE" "$AST_MIN_DELTA"
      ARGS=(--mode audio_only "${ARGS[@]}" --ast-model-name "$AST_MODEL_NAME")
      ;;
    trimodal_concat)
      batch="$(total_batch "$TRIMODAL_BATCH_PER_GPU" "$selected_gpus")"
      base_args "$save_dir" "$seed" "$selected_gpus" "$batch" "$TRIMODAL_EPOCHS" "$TRIMODAL_LR" "$TRIMODAL_PATIENCE" "$TRIMODAL_MIN_DELTA"
      ARGS=(
        --mode trimodal
        "${ARGS[@]}"
        --traj-encoder trnet_seq
        --traj-feature-map-size 6
        --image-window-size 9
        --image-sampling nearest_causal
        --image-radius 8
        --image-temporal-pool gru
        --image-temporal-delta diff
        --visual-pretrained-path "$VISUAL_PRETRAINED_PATH"
        --ast-model-name "$AST_MODEL_NAME"
        --fusion concat
        --freeze-encoders-epochs 1
        --init-traj-checkpoint "$(checkpoint_for trnet "$seed")"
        --init-image-checkpoint "$(checkpoint_for image_best "$seed")"
        --init-audio-checkpoint "$(checkpoint_for ast "$seed")"
      )
      ;;
    *)
      echo "Unknown experiment: $exp_id" >&2
      exit 2
      ;;
  esac
}

check_inputs() {
  local selected_gpus="$1"
  require_selected_4090_gpus "$selected_gpus"
  for path in "$TRAIN_CSV" "$VAL_CSV" "$TEST_CSV" "$DURATION_STATS"; do
    if [[ ! -f "$path" ]]; then
      echo "Missing required input: $path" >&2
      exit 1
    fi
  done
  if [[ ! -f "$VISUAL_PRETRAINED_PATH" ]]; then
    echo "Missing visual checkpoint: $VISUAL_PRETRAINED_PATH" >&2
    exit 1
  fi
}

run_one() {
  local exp_id="$1"
  local seed="$2"
  local selected_gpus="$3"
  local save_dir run_log cmd_file
  save_dir="$(run_dir_for "$exp_id" "$seed")"
  run_log="$LOG_DIR/${exp_id}_seed${seed}.log"
  cmd_file="$LOG_DIR/commands/${exp_id}_seed${seed}.sh"
  if [[ "$FORCE" != "1" && -f "$save_dir/summary.json" ]]; then
    echo "[$(date -Is)] SKIP exp=$exp_id seed=$seed summary exists: $save_dir/summary.json"
    return 0
  fi
  mkdir -p "$save_dir" "$LOG_DIR/commands"
  build_args_for "$exp_id" "$seed" "$selected_gpus" "$save_dir"
  CMD=(
    env -u LD_PRELOAD -u PROXYCHAINS_CONF_FILE
    CUDA_DEVICE_ORDER=PCI_BUS_ID
    "CUDA_VISIBLE_DEVICES=$selected_gpus"
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
  echo "[$(date -Is)] START exp=$exp_id seed=$seed gpus=$selected_gpus save_dir=$save_dir"
  echo "[$(date -Is)] log=$run_log cmd=$cmd_file"
  "${CMD[@]}" > "$run_log" 2>&1
  echo "[$(date -Is)] DONE exp=$exp_id seed=$seed"
}

write_env_file() {
  mkdir -p "$LOG_DIR"
  {
    printf 'export RUN_TAG=%q\n' "$RUN_TAG"
    printf 'export SUITE_DIR=%q\n' "$SUITE_DIR"
    printf 'export LOG_DIR=%q\n' "$LOG_DIR"
    printf 'export CONDA_BIN=%q\n' "$CONDA_BIN"
    printf 'export CONDA_ENV=%q\n' "$CONDA_ENV"
    printf 'export TRAIN_CSV=%q\n' "$TRAIN_CSV"
    printf 'export VAL_CSV=%q\n' "$VAL_CSV"
    printf 'export TEST_CSV=%q\n' "$TEST_CSV"
    printf 'export DURATION_STATS=%q\n' "$DURATION_STATS"
    printf 'export VISUAL_PRETRAINED_PATH=%q\n' "$VISUAL_PRETRAINED_PATH"
    printf 'export AST_MODEL_NAME=%q\n' "$AST_MODEL_NAME"
    printf 'export GPU_IDS_REQUEST=%q\n' "$GPU_IDS_REQUEST"
    printf 'export FALLBACK_GPU_IDS=%q\n' "$FALLBACK_GPU_IDS"
    printf 'export TRUST_GPU_IDS_ON_QUERY_FAIL=%q\n' "$TRUST_GPU_IDS_ON_QUERY_FAIL"
    printf 'export GPU_BUSY_MEMORY_MB=%q\n' "$GPU_BUSY_MEMORY_MB"
    printf 'export MIN_GPUS=%q\n' "$MIN_GPUS"
    printf 'export SEEDS=%q\n' "$SEEDS"
    printf 'export EXPERIMENTS=%q\n' "$EXPERIMENTS"
    printf 'export FORCE=%q\n' "$FORCE"
    printf 'export SKIP_CHECK=%q\n' "$SKIP_CHECK"
    printf 'export SEQ_LEN=%q\n' "$SEQ_LEN"
    printf 'export NUM_WORKERS=%q\n' "$NUM_WORKERS"
    printf 'export TRNET_EPOCHS=%q\n' "$TRNET_EPOCHS"
    printf 'export IMAGE_EPOCHS=%q\n' "$IMAGE_EPOCHS"
    printf 'export AST_EPOCHS=%q\n' "$AST_EPOCHS"
    printf 'export TRIMODAL_EPOCHS=%q\n' "$TRIMODAL_EPOCHS"
    printf 'export TRNET_PATIENCE=%q\n' "$TRNET_PATIENCE"
    printf 'export IMAGE_PATIENCE=%q\n' "$IMAGE_PATIENCE"
    printf 'export AST_PATIENCE=%q\n' "$AST_PATIENCE"
    printf 'export TRIMODAL_PATIENCE=%q\n' "$TRIMODAL_PATIENCE"
    printf 'export TRNET_MIN_DELTA=%q\n' "$TRNET_MIN_DELTA"
    printf 'export IMAGE_MIN_DELTA=%q\n' "$IMAGE_MIN_DELTA"
    printf 'export AST_MIN_DELTA=%q\n' "$AST_MIN_DELTA"
    printf 'export TRIMODAL_MIN_DELTA=%q\n' "$TRIMODAL_MIN_DELTA"
    printf 'export TRNET_BATCH_PER_GPU=%q\n' "$TRNET_BATCH_PER_GPU"
    printf 'export IMAGE_BATCH_PER_GPU=%q\n' "$IMAGE_BATCH_PER_GPU"
    printf 'export AST_BATCH_PER_GPU=%q\n' "$AST_BATCH_PER_GPU"
    printf 'export TRIMODAL_BATCH_PER_GPU=%q\n' "$TRIMODAL_BATCH_PER_GPU"
    printf 'export TRNET_LR=%q\n' "$TRNET_LR"
    printf 'export IMAGE_LR=%q\n' "$IMAGE_LR"
    printf 'export AST_LR=%q\n' "$AST_LR"
    printf 'export TRIMODAL_LR=%q\n' "$TRIMODAL_LR"
  } > "$ENV_FILE"
}

run_suite() {
  local selected_gpus exp_id seed status
  selected_gpus="$(select_gpu_ids)"
  if [[ "$SKIP_CHECK" != "1" ]]; then
    check_inputs "$selected_gpus"
  fi
  mkdir -p "$LOG_DIR" "$SUITE_DIR"
  echo "[$(date -Is)] RUN_TAG=$RUN_TAG"
  echo "[$(date -Is)] selected RTX 4090 GPUs=$selected_gpus"
  echo "[$(date -Is)] batch per GPU: trnet=$TRNET_BATCH_PER_GPU image=$IMAGE_BATCH_PER_GPU ast=$AST_BATCH_PER_GPU trimodal=$TRIMODAL_BATCH_PER_GPU"
  status=0
  for exp_id in $EXPERIMENTS; do
    for seed in $SEEDS; do
      if ! run_one "$exp_id" "$seed" "$selected_gpus"; then
        status=1
        echo "[$(date -Is)] FAIL exp=$exp_id seed=$seed"
      fi
      summarize_suite || true
    done
  done
  summarize_suite || true
  exit "$status"
}

summarize_suite() {
  "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
    python scripts/summarize_paper_4090_suite.py \
      --suite-dir "$SUITE_DIR" \
      --output-dir "$LOG_DIR/results"
}

smoke() {
  local selected_gpus save_dir
  selected_gpus="$(select_gpu_ids)"
  check_inputs "$selected_gpus"
  save_dir="$SUITE_DIR/smoke/seed42"
  mkdir -p "$save_dir" "$LOG_DIR"
  base_args "$save_dir" 42 "$selected_gpus" "$(total_batch 1 "$selected_gpus")" 1 0.0003 0 0.001
  CMD=(
    env -u LD_PRELOAD -u PROXYCHAINS_CONF_FILE
    CUDA_DEVICE_ORDER=PCI_BUS_ID
    "CUDA_VISIBLE_DEVICES=$selected_gpus"
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    HF_DATASETS_OFFLINE=1
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV"
    python -u src/train_ablation.py
    --mode image_only
    "${ARGS[@]}"
    --image-window-size 9
    --image-sampling nearest_causal
    --image-temporal-pool gru
    --image-temporal-delta diff
    --visual-pretrained-path "$VISUAL_PRETRAINED_PATH"
    --max-train-batches 2
    --max-eval-batches 2
  )
  echo "[$(date -Is)] smoke GPUs=$selected_gpus"
  "${CMD[@]}"
}

status_run() {
  if [[ -f "$PID_FILE" ]] && pgrep -af "run_paper_4090_suite.sh run $RUN_TAG" | awk -v pid="$(cat "$PID_FILE")" '$1 == pid { found=1 } END { exit !found }'; then
    echo "launcher running pid=$(cat "$PID_FILE")"
  elif [[ -f "$PID_FILE" ]]; then
    echo "launcher not running pid=$(cat "$PID_FILE")"
  else
    echo "no pid file: $PID_FILE"
  fi
  pgrep -af "run_paper_4090_suite.sh run $RUN_TAG|$SUITE_DIR|src/train_ablation.py" || true
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
    SKIP_CHECK=1
    write_env_file
    nohup setsid env PAPER_CONFIG_FILE="$ENV_FILE" "$SCRIPT_PATH" run "$RUN_TAG" > "$RUN_LOG" 2>&1 < /dev/null &
    echo "$!" > "$PID_FILE"
    echo "Started paper 4090 suite."
    echo "PID: $(cat "$PID_FILE")"
    echo "Log: $RUN_LOG"
    echo "Suite: $SUITE_DIR"
    ;;
  run)
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
    summarize_suite
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
