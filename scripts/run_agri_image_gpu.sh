#!/usr/bin/env bash
set -euo pipefail

cd /private/research/Agri-MBT

RUN_ID=""
GPU_IDS=""
TRAIN_CSV=""
VAL_CSV=""
TEST_CSV=""
EPOCHS=""
MAX_TRAIN_BATCHES=""
MAX_EVAL_BATCHES=""
CLASS_WEIGHT_POWER=""
TRAIN_SAMPLER=""
SAMPLER_WEIGHT_POWER=""
AUX_TARGET_CLASSES=""
AUX_LOSS_WEIGHT=""
AUX_POS_WEIGHT_POWER=""
IMAGE_RADIUS_MODE=""
IMAGE_RADIUS_DURATION_SCALE=""
IMAGE_RADIUS_CLASSES=""
IMAGE_TEMPORAL_POOL=""
IMAGE_TEMPORAL_DELTA=""
EARLY_STOP_VAL_MACRO_F1=""
EVAL_CHECKPOINT=""

usage() {
  cat >&2 <<'USAGE'
Usage: scripts/run_agri_image_gpu.sh --run-id ID [--gpu-ids IDS] [--train-csv PATH] [--val-csv PATH] [--test-csv PATH] [--epochs N] [--max-train-batches N] [--max-eval-batches N] [--class-weight-power X] [--train-sampler shuffle|class_balanced] [--sampler-weight-power X] [--aux-target-classes IDS] [--aux-loss-weight X] [--aux-pos-weight-power X] [--image-radius-mode fixed|duration] [--image-radius-duration-scale X] [--image-radius-classes IDS] [--image-temporal-pool mean|transformer|gru] [--image-temporal-delta none|diff] [--early-stop-val-macro-f1 X] [--eval-checkpoint PATH]

Runs the image-only autoresearch workload on physical GPUs 0,1,2,5.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="${2:?missing value for --run-id}"
      shift 2
      ;;
    --gpu-ids)
      GPU_IDS="${2:?missing value for --gpu-ids}"
      shift 2
      ;;
    --train-csv)
      TRAIN_CSV="${2:?missing value for --train-csv}"
      shift 2
      ;;
    --val-csv)
      VAL_CSV="${2:?missing value for --val-csv}"
      shift 2
      ;;
    --test-csv)
      TEST_CSV="${2:?missing value for --test-csv}"
      shift 2
      ;;
    --epochs)
      EPOCHS="${2:?missing value for --epochs}"
      shift 2
      ;;
    --max-train-batches)
      MAX_TRAIN_BATCHES="${2:?missing value for --max-train-batches}"
      shift 2
      ;;
    --max-eval-batches)
      MAX_EVAL_BATCHES="${2:?missing value for --max-eval-batches}"
      shift 2
      ;;
    --class-weight-power)
      CLASS_WEIGHT_POWER="${2:?missing value for --class-weight-power}"
      shift 2
      ;;
    --train-sampler)
      TRAIN_SAMPLER="${2:?missing value for --train-sampler}"
      shift 2
      ;;
    --sampler-weight-power)
      SAMPLER_WEIGHT_POWER="${2:?missing value for --sampler-weight-power}"
      shift 2
      ;;
    --aux-target-classes)
      AUX_TARGET_CLASSES="${2:?missing value for --aux-target-classes}"
      shift 2
      ;;
    --aux-loss-weight)
      AUX_LOSS_WEIGHT="${2:?missing value for --aux-loss-weight}"
      shift 2
      ;;
    --aux-pos-weight-power)
      AUX_POS_WEIGHT_POWER="${2:?missing value for --aux-pos-weight-power}"
      shift 2
      ;;
    --image-radius-mode)
      IMAGE_RADIUS_MODE="${2:?missing value for --image-radius-mode}"
      shift 2
      ;;
    --image-radius-duration-scale)
      IMAGE_RADIUS_DURATION_SCALE="${2:?missing value for --image-radius-duration-scale}"
      shift 2
      ;;
    --image-radius-classes)
      IMAGE_RADIUS_CLASSES="${2:?missing value for --image-radius-classes}"
      shift 2
      ;;
    --image-temporal-pool)
      IMAGE_TEMPORAL_POOL="${2:?missing value for --image-temporal-pool}"
      shift 2
      ;;
    --image-temporal-delta)
      IMAGE_TEMPORAL_DELTA="${2:?missing value for --image-temporal-delta}"
      shift 2
      ;;
    --early-stop-val-macro-f1)
      EARLY_STOP_VAL_MACRO_F1="${2:?missing value for --early-stop-val-macro-f1}"
      shift 2
      ;;
    --eval-checkpoint)
      EVAL_CHECKPOINT="${2:?missing value for --eval-checkpoint}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$RUN_ID" ]]; then
  echo "--run-id is required" >&2
  usage
  exit 2
fi

export AGRI_IMAGE_GPU_IDS="${GPU_IDS:-0,1,2,5}"
export AGRI_IMAGE_RUN_ID="$RUN_ID"

if [[ -n "$TRAIN_CSV" ]]; then
  export AGRI_IMAGE_TRAIN_CSV="$TRAIN_CSV"
fi
if [[ -n "$VAL_CSV" ]]; then
  export AGRI_IMAGE_VAL_CSV="$VAL_CSV"
fi
if [[ -n "$TEST_CSV" ]]; then
  export AGRI_IMAGE_TEST_CSV="$TEST_CSV"
fi
if [[ -n "$EPOCHS" ]]; then
  export AGRI_IMAGE_EPOCHS="$EPOCHS"
fi
if [[ -n "$MAX_TRAIN_BATCHES" ]]; then
  export AGRI_IMAGE_MAX_TRAIN_BATCHES="$MAX_TRAIN_BATCHES"
fi
if [[ -n "$MAX_EVAL_BATCHES" ]]; then
  export AGRI_IMAGE_MAX_EVAL_BATCHES="$MAX_EVAL_BATCHES"
fi
if [[ -n "$CLASS_WEIGHT_POWER" ]]; then
  export AGRI_IMAGE_CLASS_WEIGHT_POWER="$CLASS_WEIGHT_POWER"
fi
if [[ -n "$TRAIN_SAMPLER" ]]; then
  export AGRI_IMAGE_TRAIN_SAMPLER="$TRAIN_SAMPLER"
fi
if [[ -n "$SAMPLER_WEIGHT_POWER" ]]; then
  export AGRI_IMAGE_SAMPLER_WEIGHT_POWER="$SAMPLER_WEIGHT_POWER"
fi
if [[ -n "$AUX_TARGET_CLASSES" ]]; then
  export AGRI_IMAGE_AUX_TARGET_CLASSES="$AUX_TARGET_CLASSES"
fi
if [[ -n "$AUX_LOSS_WEIGHT" ]]; then
  export AGRI_IMAGE_AUX_LOSS_WEIGHT="$AUX_LOSS_WEIGHT"
fi
if [[ -n "$AUX_POS_WEIGHT_POWER" ]]; then
  export AGRI_IMAGE_AUX_POS_WEIGHT_POWER="$AUX_POS_WEIGHT_POWER"
fi
if [[ -n "$IMAGE_RADIUS_MODE" ]]; then
  export AGRI_IMAGE_RADIUS_MODE="$IMAGE_RADIUS_MODE"
fi
if [[ -n "$IMAGE_RADIUS_DURATION_SCALE" ]]; then
  export AGRI_IMAGE_RADIUS_DURATION_SCALE="$IMAGE_RADIUS_DURATION_SCALE"
fi
if [[ -n "$IMAGE_RADIUS_CLASSES" ]]; then
  export AGRI_IMAGE_RADIUS_CLASSES="$IMAGE_RADIUS_CLASSES"
fi
if [[ -n "$IMAGE_TEMPORAL_POOL" ]]; then
  export AGRI_IMAGE_TEMPORAL_POOL="$IMAGE_TEMPORAL_POOL"
fi
if [[ -n "$IMAGE_TEMPORAL_DELTA" ]]; then
  export AGRI_IMAGE_TEMPORAL_DELTA="$IMAGE_TEMPORAL_DELTA"
fi
if [[ -n "$EARLY_STOP_VAL_MACRO_F1" ]]; then
  export AGRI_IMAGE_EARLY_STOP_VAL_MACRO_F1="$EARLY_STOP_VAL_MACRO_F1"
fi
if [[ -n "$EVAL_CHECKPOINT" ]]; then
  export AGRI_IMAGE_EVAL_CHECKPOINT="$EVAL_CHECKPOINT"
fi

exec ./autoresearch.sh
