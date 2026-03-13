#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Loader for Trajectory Classification
Adapted for 11-class agricultural activity classification
"""

import os
import torch
from torch.utils.data import DataLoader
import torch.utils.data as Data
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
import opt

device = opt.device


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
        X_train, y_train, X_valid, y_valid, X_test, y_test, X_all, y_all
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
    for train_index, test_index in sss.split(train_data, train_tag):
        X_train, X_test = train_data[train_index], train_data[test_index]
        y_train, y_test = train_tag[train_index], train_tag[test_index]

    # Split train into train/valid
    sss = ShuffleSplit(n_splits=1, test_size=opt.valRatio, random_state=42)
    for train_index, test_index in sss.split(X_train, y_train):
        X_train, X_valid = X_train[train_index], X_train[test_index]
        y_train, y_valid = y_train[train_index], y_train[test_index]

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

    return X_train, y_train, X_valid, y_valid, X_test, y_test, X_all, y_all


def get_loader_trajectory(path, step=8, batch_size=64, num_workers=4):
    """
    Get data loaders for trajectory-only training

    Returns:
        train_loader, valid_loader, test_loader, all_loader
    """
    X_train, y_train, X_valid, y_valid, X_test, y_test, X_all, y_all = get_data_trajectory(path, step)

    train_dataset = Data.TensorDataset(X_train, y_train)
    valid_dataset = Data.TensorDataset(X_valid, y_valid)
    test_dataset = Data.TensorDataset(X_test, y_test)
    all_dataset = Data.TensorDataset(X_all, y_all)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader, all_loader


def get_data_multimodal(path, step=8):
    """
    Load multimodal data (trajectory + image features) for BiLSTM

    Returns:
        X_traj_train, X_img_train, y_train, ...
    """
    # Load trajectory data
    X_train, y_train, X_valid, y_valid, X_test, y_test, X_all, y_all = get_data_trajectory(path, step)

    # Load image features (from MBT model or pre-extracted features)
    # For now, we'll use placeholder - in practice, you'd load pre-extracted features
    # from a ViT or other vision model

    # Placeholder: random image features (replace with actual features)
    img_feat_size = 768  # ViT-B16 feature dimension

    def generate_img_features(n_samples):
        """Placeholder: generate random image features"""
        return torch.randn(n_samples, img_feat_size).to(device)

    X_img_train = generate_img_features(X_train.shape[0])
    X_img_valid = generate_img_features(X_valid.shape[0])
    X_img_test = generate_img_features(X_test.shape[0])
    X_img_all = generate_img_features(X_all.shape[0])

    return X_train, X_img_train, y_train, \
           X_valid, X_img_valid, y_valid, \
           X_test, X_img_test, y_test, \
           X_all, X_img_all, y_all


def get_loader_multimodal(path, step=8, batch_size=64, num_workers=4):
    """
    Get data loaders for multimodal training (trajectory + image)

    Returns:
        train_loader, valid_loader, test_loader, all_loader
    """
    X_traj_train, X_img_train, y_train, \
    X_traj_valid, X_img_valid, y_valid, \
    X_traj_test, X_img_test, y_test, \
    X_traj_all, X_img_all, y_all = get_data_multimodal(path, step)

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
