#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Loader for Trajectory Classification
Adapted for 11-class agricultural activity classification
"""

import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.utils.data as Data
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
import opt

device = opt.device
DEFAULT_VISUAL_FEATURES_PATH = Path(__file__).resolve().parents[1] / "data" / "visual_features.npz"


def transform_dataset(x_data, y_data, n_input, n_output=1):
    """
    Transform dataset into sequence format for LSTM

    Args:
        x_data: (n_samples, n_features)
        y_data: (n_samples, 1)
        n_input: sequence length
        n_output: fixed to 1

    Returns:
        X: (n_samples - n_input + 1, n_input, n_features)
        Y: (n_samples - n_input + 1, 1)
    """
    all_data = x_data
    data_size = x_data.shape[0]
    X = np.empty((data_size - n_input + 1, n_input, all_data.shape[1]))
    Y = np.empty((data_size - n_input + 1, y_data.shape[1]))

    for i in range(data_size - n_input + 1):
        X[i] = all_data[i:i + n_input, :]
        Y[i] = y_data[i + n_input - 1, :]

    return X, Y


def transform_indices(indices, n_input):
    """
    Align per-row indices with sequence labels by using the last row in each window.
    """
    return indices[n_input - 1:]


# Trajectory feature columns (actual columns from aligned_data.csv)
TRAJ_COLS = [
    '经度', '纬度', '间距(米)', '深度', '速度', '方向角'
]


def get_data_trajectory(path, step=8):
    """
    Load trajectory data for BiLSTM

    Args:
        path: path to aligned_data.csv
        step: sequence length (default 8 frames)

    Returns:
        X_train, y_train, X_valid, y_valid, X_test, y_test, X_all, y_all,
        train_indices, valid_indices, test_indices, all_indices
    """
    # Load data
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} samples from {path}")

    # Extract trajectory features (only columns that exist)
    available_cols = [col for col in TRAJ_COLS if col in df.columns]
    print(f"Using {len(available_cols)} trajectory features: {available_cols}")

    train_data = df[available_cols].values.astype(float)
    train_tag = df['分类'].values.astype(int).reshape(-1, 1)

    # Split into train/test
    sss = ShuffleSplit(n_splits=1, test_size=opt.testRatio, random_state=42)
    train_valid_indices, test_indices = next(sss.split(train_data, train_tag))
    X_train_valid, X_test = train_data[train_valid_indices], train_data[test_indices]
    y_train_valid, y_test = train_tag[train_valid_indices], train_tag[test_indices]

    # Split train into train/valid
    sss = ShuffleSplit(n_splits=1, test_size=opt.valRatio, random_state=42)
    train_sub_indices, valid_sub_indices = next(sss.split(X_train_valid, y_train_valid))
    train_indices = train_valid_indices[train_sub_indices]
    valid_indices = train_valid_indices[valid_sub_indices]

    X_train, X_valid = X_train_valid[train_sub_indices], X_train_valid[valid_sub_indices]
    y_train, y_valid = y_train_valid[train_sub_indices], y_train_valid[valid_sub_indices]

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)
    X_all = scaler.transform(train_data)

    # Transform to sequences
    X_train, y_train = transform_dataset(X_train, y_train, step, 1)
    X_valid, y_valid = transform_dataset(X_valid, y_valid, step, 1)
    X_test, y_test = transform_dataset(X_test, y_test, step, 1)
    X_all, y_all = transform_dataset(X_all, train_tag, step, 1)
    train_indices = transform_indices(train_indices, step)
    valid_indices = transform_indices(valid_indices, step)
    test_indices = transform_indices(test_indices, step)
    all_indices = transform_indices(np.arange(len(train_data)), step)

    print(f"Train: {X_train.shape[0]} sequences, Valid: {X_valid.shape[0]}, Test: {X_test.shape[0]}")

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).squeeze(1).to(device)
    X_valid = torch.tensor(X_valid, dtype=torch.float32).to(device)
    y_valid = torch.tensor(y_valid, dtype=torch.long).squeeze(1).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).squeeze(1).to(device)
    X_all = torch.tensor(X_all, dtype=torch.float32).to(device)
    y_all = torch.tensor(y_all, dtype=torch.long).squeeze(1).to(device)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, X_all, y_all, \
           train_indices, valid_indices, test_indices, all_indices


def get_loader_trajectory(path, step=8, batch_size=64, num_workers=4):
    """
    Get data loaders for trajectory-only training

    Returns:
        train_loader, valid_loader, test_loader, all_loader
    """
    X_train, y_train, X_valid, y_valid, X_test, y_test, X_all, y_all, *_ = get_data_trajectory(path, step)

    train_dataset = Data.TensorDataset(X_train, y_train)
    valid_dataset = Data.TensorDataset(X_valid, y_valid)
    test_dataset = Data.TensorDataset(X_test, y_test)
    all_dataset = Data.TensorDataset(X_all, y_all)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader, all_loader


def get_data_multimodal(path, step=8, visual_features_path=None):
    """
    Load multimodal data (trajectory + image features) for BiLSTM

    Returns:
        X_traj_train, X_img_train, y_train, ...
    """
    # Load trajectory data
    X_train, y_train, X_valid, y_valid, X_test, y_test, X_all, y_all, \
    train_indices, valid_indices, test_indices, all_indices = get_data_trajectory(path, step)

    features_path = Path(visual_features_path) if visual_features_path else DEFAULT_VISUAL_FEATURES_PATH
    if not features_path.exists():
        raise NotImplementedError(
            f"Visual features file not found at {features_path}. "
            "Run extract_visual_features.py first to generate BiLSTM-trajectory/data/visual_features.npz."
        )

    features_data = np.load(features_path)
    if 'features' not in features_data:
        raise KeyError(f"Missing 'features' array in {features_path}")

    all_img_features = features_data['features']
    if all_img_features.ndim != 2 or all_img_features.shape[1] != 768:
        raise ValueError(
            f"Expected visual features with shape (N, 768), got {all_img_features.shape} from {features_path}"
        )

    df = pd.read_csv(path)
    if all_img_features.shape[0] != len(df):
        raise ValueError(
            f"Visual feature count ({all_img_features.shape[0]}) does not match CSV rows ({len(df)}). "
            "Re-run extract_visual_features.py on the same aligned_data.csv file used for training."
        )

    all_img_features = torch.tensor(all_img_features, dtype=torch.float32).to(device)
    train_indices = torch.as_tensor(train_indices, dtype=torch.long, device=device)
    valid_indices = torch.as_tensor(valid_indices, dtype=torch.long, device=device)
    test_indices = torch.as_tensor(test_indices, dtype=torch.long, device=device)
    all_indices = torch.as_tensor(all_indices, dtype=torch.long, device=device)

    X_img_train = all_img_features[train_indices]
    X_img_valid = all_img_features[valid_indices]
    X_img_test = all_img_features[test_indices]
    X_img_all = all_img_features[all_indices]

    return X_train, X_img_train, y_train, \
           X_valid, X_img_valid, y_valid, \
           X_test, X_img_test, y_test, \
           X_all, X_img_all, y_all


def get_loader_multimodal(path, step=8, batch_size=64, num_workers=4, visual_features_path=None):
    """
    Get data loaders for multimodal training (trajectory + image)

    Returns:
        train_loader, valid_loader, test_loader, all_loader
    """
    X_traj_train, X_img_train, y_train, \
    X_traj_valid, X_img_valid, y_valid, \
    X_traj_test, X_img_test, y_test, \
    X_traj_all, X_img_all, y_all = get_data_multimodal(path, step, visual_features_path=visual_features_path)

    train_dataset = Data.TensorDataset(X_traj_train, X_img_train, y_train)
    valid_dataset = Data.TensorDataset(X_traj_valid, X_img_valid, y_valid)
    test_dataset = Data.TensorDataset(X_traj_test, X_img_test, y_test)
    all_dataset = Data.TensorDataset(X_traj_all, X_img_all, y_all)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader, all_loader


if __name__ == "__main__":
    # Test data loading
    path = "/home/research/Agri-MBT/data/aligned_output/aligned_data.csv"
    train_loader, valid_loader, test_loader, _ = get_loader_trajectory(path, step=8, batch_size=32)

    print(f"\nData loader test:")
    for X, y in train_loader:
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  y unique values: {torch.unique(y)}")
        break
