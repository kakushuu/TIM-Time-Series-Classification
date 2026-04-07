#!/bin/bash
# Hierarchical BiLSTM training script

PYTHONUNBUFFERED=1 python -u /home/research/Agri-MBT/Multimodal-Fusion-with-Attention-Bottlenecks-main/MBT/train_test.py \
  --mode trajectory_only \
  --traj_arch hierarchical_bilstm \
  --num_epochs 20 \
  --batch_size 8 \
  --lr 3e-4 \
  --loss_type weighted_ce \
  --csv_file /home/research/Agri-MBT/data/aligned_output/aligned_data_27features.csv \
  --data_dir /home/research/Agri-MBT \
  --output_dir /home/research/Agri-MBT/experiments \
  2>&1 | tee /home/research/Agri-MBT/experiments/train_hierarchical_bilstm.log
