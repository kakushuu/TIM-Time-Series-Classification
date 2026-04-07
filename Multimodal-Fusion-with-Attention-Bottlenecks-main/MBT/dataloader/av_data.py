import os
import pandas as pd
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as Tv

# 27 trajectory feature columns (仿照 GAN-BiLSTM)
TRAJ_COLS = [
    '经度', '纬度',  # 2 维经纬度
    # 5 个运动特征 × 5 个统计量 = 25 维
    '速度', 'speed_med_5', 'speed_med_50', 'speed_SD_5', 'speed_SD_50',
    'acceleration', 'acceleration_med_5', 'acceleration_med_50', 'acceleration_SD_5', 'acceleration_SD_50',
    'angular_speed', 'angular_speed_med_5', 'angular_speed_med_50', 'angular_speed_SD_5', 'angular_speed_SD_50',
    'angular_acceleration', 'angular_acceleration_med_5', 'angular_acceleration_med_50', 'angular_acceleration_SD_5', 'angular_acceleration_SD_50',
    'angle_diff', 'angle_diff_med_5', 'angle_diff_med_50', 'angle_diff_SD_5', 'angle_diff_SD_50'
]
TRAJ_SEQ  = 512  # sliding-window length (仿照 GAN-BiLSTM)


class AV_Dataset(Dataset):
    def __init__(self, df, data_dir='', traj_mean=None, traj_std=None, seq_len=TRAJ_SEQ, load_images=True):
        """
        Args:
            df:        DataFrame with 27 trajectory features + 分类 + frame_path
            data_dir:  Base directory prepended to frame_path (leave '' if paths are already absolute)
            traj_mean: Pre-computed mean for trajectory normalisation (numpy array, shape (27,))
            traj_std:  Pre-computed std  for trajectory normalisation (numpy array, shape (27,))
            seq_len:   Sliding-window length T; each sample returns T consecutive trajectory frames (default 512)
            load_images: If False, skip image loading (for trajectory_only mode, speeds up data loading)
        """
        super(AV_Dataset, self).__init__()

        self.df       = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.seq_len  = seq_len
        self.load_images = load_images

        if load_images:
            self.visual_transforms = Tv.Compose([
                Tv.Resize((224, 224)),
                Tv.ToTensor(),
                Tv.ConvertImageDtype(torch.float32),
                Tv.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        # Trajectory normalisation stats
        if traj_mean is None:
            traj = self.df[TRAJ_COLS].values.astype(np.float32)
            traj_mean = traj.mean(axis=0)
            traj_std  = traj.std(axis=0) + 1e-6
        self.traj_mean = torch.tensor(traj_mean, dtype=torch.float32)
        self.traj_std  = torch.tensor(traj_std,  dtype=torch.float32)

        # Pre-load all trajectory features as a tensor for fast window slicing
        traj_vals = self.df[TRAJ_COLS].values.astype(np.float32)          # (N, 27)
        traj_vals = torch.tensor(traj_vals)                                # (N, 27)
        self.traj_all = (traj_vals - self.traj_mean) / self.traj_std       # normalised

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Image (current frame only) ──────────────────────────────────────
        if self.load_images:
            frame_path = os.path.join(self.data_dir, row['frame_path']) if self.data_dir else row['frame_path']
            img = Image.open(frame_path).convert('RGB')
            img = self.visual_transforms(img)   # (3, 224, 224)
            rgb_frames = img.unsqueeze(0)        # (1, 3, 224, 224)
        else:
            # For trajectory_only mode, return dummy tensor (will be ignored)
            rgb_frames = torch.zeros(1, 3, 224, 224)

        # ── Trajectory sequence: sliding window [idx-T+1 … idx] ─────────────
        # If idx < T-1 we repeat the earliest available frame (left-padding).
        start = max(0, idx - self.seq_len + 1)
        traj_seq = self.traj_all[start : idx + 1]           # (<=T, 27)

        if traj_seq.shape[0] < self.seq_len:
            # Pad on the left by repeating the first row
            pad = traj_seq[0:1].expand(self.seq_len - traj_seq.shape[0], -1)
            traj_seq = torch.cat([pad, traj_seq], dim=0)    # (T, 27)

        # traj_seq: (T, 27)

        # ── Label ────────────────────────────────────────────────────────────
        label = int(row['分类'])

        return traj_seq, rgb_frames, label
