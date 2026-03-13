#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration for BiLSTM Trajectory Classification
Adapted for 11-class agricultural activity classification
"""

import torch
import os

# Project paths
PROJECT_ROOT = "/home/research/Agri-MBT/BiLSTM-trajectory"
DATA_ROOT = "/home/research/Agri-MBT/data/aligned_output"

# Model parameters
n_classes = 11  # 11 agricultural activity classes
emb_size = 6  # 6 trajectory features (经度,纬度,间距,深度,速度,方向角)
rnn_size = 256
rnn_layers = 2
dropout = 0.3

# Training parameters
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
ALPHA = 0.75
GAMMA = 2

batch_size = 64
epochs = 50
number_workers = 0  # Set to 0 to avoid CUDA initialization error
accumulation_step = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
optimizer = "adamw"
lrsc = "warmup"  # "warmup" or "plateau"
loss = "Focal"  # "Focal" or "CE"

# Sequence length (8 frames per clip, matching MBT model)
time_tri = 8

# Data paths
data_dir = os.path.join(DATA_ROOT, "aligned_data.csv")
save_dir = os.path.join(PROJECT_ROOT, "experiments", "results")
filename = "bilstm_trajectory_11class"

# Model control
keep = True
resume = False
resume_path = os.path.join(PROJECT_ROOT, "weights", "bilstm_trajectory.pth")

# GAN settings (disabled)
useGAN = False
useGAN_weights = False
GAN_path = ""
GAN_epoch = 0

# Model name
NAME = "BiLSTM_Trajectory"

# Data split
testRatio = 0.2
valRatio = 0.2

# Mode: "trajectory_only" or "multimodal"
mode = "trajectory_only"
