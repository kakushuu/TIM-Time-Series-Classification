#!/usr/bin/env bash
set -euo pipefail

cd /private/research/Agri-MBT

RUN_ID=""
MODE=""
GPU_IDS=""
TRAIN_CSV=""
VAL_CSV=""
TEST_CSV=""
EPOCHS=""
MAX_TRAIN_BATCHES=""
MAX_EVAL_BATCHES=""
CLASS_WEIGHT_POWER=""
LOSS_TYPE=""
FOCAL_GAMMA=""
CB_BETA=""
LABEL_SMOOTHING=""
TRAIN_SAMPLER=""
SAMPLER_WEIGHT_POWER=""
SAMPLER_BOOST_CLASSES=""
SAMPLER_BOOST_FACTOR=""
AUX_TARGET_CLASSES=""
AUX_LOSS_WEIGHT=""
AUX_POS_WEIGHT_POWER=""
IMAGE_RADIUS_MODE=""
IMAGE_RADIUS_DURATION_SCALE=""
IMAGE_RADIUS_CLASSES=""
IMAGE_TEMPORAL_POOL=""
IMAGE_TEMPORAL_DELTA=""
FEATURE_MODE=""
TRAJ_ENCODER=""
TRAJ_FEATURE_MAP_SIZE=""
PRETRAINED=""
VISUAL_PRETRAINED_PATH=""
MAX_TIME_GAP=""
EARLY_STOP_VAL_MACRO_F1=""
EARLY_STOP_PATIENCE=""
EARLY_STOP_MIN_DELTA=""
TEMPORAL_SMOOTHING=""
SMOOTH_CLASSES=""
SMOOTH_MIN_DURATION=""
EVAL_CHECKPOINT=""

usage() {
  cat >&2 <<'USAGE'
Usage: scripts/run_agri_image_gpu.sh --run-id ID [--mode image_only|trajectory_only|multimodal] [--gpu-ids IDS] [--train-csv PATH] [--val-csv PATH] [--test-csv PATH] [--epochs N] [--max-train-batches N] [--max-eval-batches N] [--class-weight-power X] [--loss-type weighted_ce|focal|cb_focal] [--label-smoothing X] [--train-sampler shuffle|class_balanced|class_boost] [--sampler-weight-power X] [--sampler-boost-classes IDS] [--sampler-boost-factor X] [--aux-target-classes IDS] [--aux-loss-weight X] [--aux-pos-weight-power X] [--image-radius-mode fixed|duration] [--image-radius-duration-scale X] [--image-radius-classes IDS] [--image-temporal-pool mean|transformer|gru] [--image-temporal-delta none|diff] [--feature-mode raw|engineered] [--traj-encoder lstm|atrnet|trnet_seq] [--traj-feature-map-size N] [--pretrained|--no-pretrained] [--visual-pretrained-path PATH] [--max-time-gap SECONDS] [--early-stop-val-macro-f1 X] [--early-stop-patience N] [--temporal-smoothing none|min_duration] [--smooth-classes IDS] [--smooth-min-duration N] [--eval-checkpoint PATH]

Runs the image-only autoresearch workload on physical GPUs 0,1,2,5.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="${2:?missing value for --run-id}"
      shift 2
      ;;
    --mode)
      MODE="${2:?missing value for --mode}"
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
    --loss-type)
      LOSS_TYPE="${2:?missing value for --loss-type}"
      shift 2
      ;;
    --focal-gamma)
      FOCAL_GAMMA="${2:?missing value for --focal-gamma}"
      shift 2
      ;;
    --cb-beta)
      CB_BETA="${2:?missing value for --cb-beta}"
      shift 2
      ;;
    --label-smoothing)
      LABEL_SMOOTHING="${2:?missing value for --label-smoothing}"
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
    --sampler-boost-classes)
      SAMPLER_BOOST_CLASSES="${2:?missing value for --sampler-boost-classes}"
      shift 2
      ;;
    --sampler-boost-factor)
      SAMPLER_BOOST_FACTOR="${2:?missing value for --sampler-boost-factor}"
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
    --feature-mode)
      FEATURE_MODE="${2:?missing value for --feature-mode}"
      shift 2
      ;;
    --traj-encoder)
      TRAJ_ENCODER="${2:?missing value for --traj-encoder}"
      shift 2
      ;;
    --traj-feature-map-size)
      TRAJ_FEATURE_MAP_SIZE="${2:?missing value for --traj-feature-map-size}"
      shift 2
      ;;
    --pretrained)
      PRETRAINED="true"
      shift
      ;;
    --no-pretrained)
      PRETRAINED="false"
      shift
      ;;
    --visual-pretrained-path)
      VISUAL_PRETRAINED_PATH="${2:?missing value for --visual-pretrained-path}"
      shift 2
      ;;
    --max-time-gap)
      MAX_TIME_GAP="${2:?missing value for --max-time-gap}"
      shift 2
      ;;
    --early-stop-val-macro-f1)
      EARLY_STOP_VAL_MACRO_F1="${2:?missing value for --early-stop-val-macro-f1}"
      shift 2
      ;;
    --early-stop-patience)
      EARLY_STOP_PATIENCE="${2:?missing value for --early-stop-patience}"
      shift 2
      ;;
    --early-stop-min-delta)
      EARLY_STOP_MIN_DELTA="${2:?missing value for --early-stop-min-delta}"
      shift 2
      ;;
    --temporal-smoothing)
      TEMPORAL_SMOOTHING="${2:?missing value for --temporal-smoothing}"
      shift 2
      ;;
    --smooth-classes)
      SMOOTH_CLASSES="${2:?missing value for --smooth-classes}"
      shift 2
      ;;
    --smooth-min-duration)
      SMOOTH_MIN_DURATION="${2:?missing value for --smooth-min-duration}"
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

if [[ -n "$MODE" ]]; then
  export AGRI_IMAGE_MODE="$MODE"
fi
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
if [[ -n "$LOSS_TYPE" ]]; then
  export AGRI_IMAGE_LOSS_TYPE="$LOSS_TYPE"
fi
if [[ -n "$FOCAL_GAMMA" ]]; then
  export AGRI_IMAGE_FOCAL_GAMMA="$FOCAL_GAMMA"
fi
if [[ -n "$CB_BETA" ]]; then
  export AGRI_IMAGE_CB_BETA="$CB_BETA"
fi
if [[ -n "$LABEL_SMOOTHING" ]]; then
  export AGRI_IMAGE_LABEL_SMOOTHING="$LABEL_SMOOTHING"
fi
if [[ -n "$TRAIN_SAMPLER" ]]; then
  export AGRI_IMAGE_TRAIN_SAMPLER="$TRAIN_SAMPLER"
fi
if [[ -n "$SAMPLER_WEIGHT_POWER" ]]; then
  export AGRI_IMAGE_SAMPLER_WEIGHT_POWER="$SAMPLER_WEIGHT_POWER"
fi
if [[ -n "$SAMPLER_BOOST_CLASSES" ]]; then
  export AGRI_IMAGE_SAMPLER_BOOST_CLASSES="$SAMPLER_BOOST_CLASSES"
fi
if [[ -n "$SAMPLER_BOOST_FACTOR" ]]; then
  export AGRI_IMAGE_SAMPLER_BOOST_FACTOR="$SAMPLER_BOOST_FACTOR"
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
if [[ -n "$FEATURE_MODE" ]]; then
  export AGRI_IMAGE_FEATURE_MODE="$FEATURE_MODE"
fi
if [[ -n "$TRAJ_ENCODER" ]]; then
  export AGRI_IMAGE_TRAJ_ENCODER="$TRAJ_ENCODER"
fi
if [[ -n "$TRAJ_FEATURE_MAP_SIZE" ]]; then
  export AGRI_IMAGE_TRAJ_FEATURE_MAP_SIZE="$TRAJ_FEATURE_MAP_SIZE"
fi
if [[ -n "$PRETRAINED" ]]; then
  export AGRI_IMAGE_PRETRAINED="$PRETRAINED"
fi
if [[ -n "$VISUAL_PRETRAINED_PATH" ]]; then
  export AGRI_IMAGE_VISUAL_PRETRAINED_PATH="$VISUAL_PRETRAINED_PATH"
fi
if [[ -n "$MAX_TIME_GAP" ]]; then
  export AGRI_IMAGE_MAX_TIME_GAP="$MAX_TIME_GAP"
fi
if [[ -n "$EARLY_STOP_VAL_MACRO_F1" ]]; then
  export AGRI_IMAGE_EARLY_STOP_VAL_MACRO_F1="$EARLY_STOP_VAL_MACRO_F1"
fi
if [[ -n "$EARLY_STOP_PATIENCE" ]]; then
  export AGRI_IMAGE_EARLY_STOP_PATIENCE="$EARLY_STOP_PATIENCE"
fi
if [[ -n "$EARLY_STOP_MIN_DELTA" ]]; then
  export AGRI_IMAGE_EARLY_STOP_MIN_DELTA="$EARLY_STOP_MIN_DELTA"
fi
if [[ -n "$TEMPORAL_SMOOTHING" ]]; then
  export AGRI_IMAGE_TEMPORAL_SMOOTHING="$TEMPORAL_SMOOTHING"
fi
if [[ -n "$SMOOTH_CLASSES" ]]; then
  export AGRI_IMAGE_SMOOTH_CLASSES="$SMOOTH_CLASSES"
fi
if [[ -n "$SMOOTH_MIN_DURATION" ]]; then
  export AGRI_IMAGE_SMOOTH_MIN_DURATION="$SMOOTH_MIN_DURATION"
fi
if [[ -n "$EVAL_CHECKPOINT" ]]; then
  export AGRI_IMAGE_EVAL_CHECKPOINT="$EVAL_CHECKPOINT"
fi

exec ./autoresearch.sh
