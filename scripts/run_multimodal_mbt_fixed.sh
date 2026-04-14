#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,5,6}" \
/private/miniforge3/bin/conda run --no-capture-output -n agri-mbt \
python src/train_ablation.py \
  --mode multimodal \
  --train-csv data/taif_20241018_split/train.csv \
  --val-csv data/taif_20241018_split/val.csv \
  --test-csv data/taif_20241018_split/test.csv \
  --save-dir experiments/new_adaptive_mbt_20241018_full/multimodal_mbt_fixed \
  --seq-len 512 \
  --stride 20 \
  --eval-stride 1 \
  --context-mode causal \
  --sampling-strategy adaptive \
  --duration-stats experiments/new_adaptive_mbt_20241018_full/behavior_duration_analysis/duration_sampling_config.json \
  --image-window-size 5 \
  --image-sampling center \
  --image-radius 4 \
  --fusion mbt \
  --fusion-layers 2 \
  --fusion-heads 8 \
  --num-latents 4 \
  --pretrained \
  --init-traj-checkpoint experiments/new_adaptive_mbt_20241018_full/trajectory_only/best.pt \
  --init-image-checkpoint experiments/new_adaptive_mbt_20241018_full/image_only/best.pt \
  --freeze-encoders-epochs 3 \
  --lr 1e-4 \
  --grad-clip 1.0 \
  --epochs 30 \
  --batch-size 4 \
  --num-workers 8 \
  --device cuda \
  --all-gpus
