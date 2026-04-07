#!/usr/bin/env python3
"""
Train trajectory-only BiLSTM with SMOTE oversampling for rare classes.

SMOTE generates synthetic samples for minority classes to balance the training set.
"""

import argparse
import numpy as np
import pandas as pd
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE

# Import from the main training script
import sys
sys.path.insert(0, '/home/research/Agri-MBT/Multimodal-Fusion-with-Attention-Bottlenecks-main/MBT')
from models.visual_model import AVmodel
from dataloader.av_data import TRAJ_COLS


def parse_args():
    parser = argparse.ArgumentParser(description="Train trajectory-only BiLSTM with SMOTE")
    parser.add_argument('--csv_file', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='/home/research/Agri-MBT')
    parser.add_argument('--output_dir', type=str, default='/home/research/Agri-MBT/experiments')
    parser.add_argument('--smote_strategy', type=str, default='auto',
                       help='SMOTE sampling strategy: "auto" or dict like {1: 2000, 6: 2000}')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Experiment: trajectory_only with SMOTE oversampling")
    print("=" * 70)

    # Load data
    df = pd.read_csv(args.csv_file)
    print(f"\nTotal samples: {len(df)}")
    print("Class distribution:")
    print(df['分类'].value_counts().sort_index())

    # Extract trajectory features and labels
    traj_cols = [c for c in df.columns if c in TRAJ_COLS]
    X = df[traj_cols].values  # (N, 27)
    y = df['分类'].values     # (N,)

    # Filter out samples with missing frames (same as original script)
    valid_mask = df['frame_path'].apply(lambda p: os.path.exists(os.path.join(args.data_dir, p)))
    X = X[valid_mask]
    y = y[valid_mask]
    print(f"\nSkipping {len(df) - len(X)} rows with missing frame files")
    print(f"Remaining samples: {len(X)}")

    # Train/test split (stratified)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\tDataset split — train: {len(X_train)}, test: {len(X_test)}")

    # Apply SMOTE to training data only
    print(f"\n🔄 Applying SMOTE oversampling...")
    print(f"Before SMOTE: {len(X_train)} samples")
    print(f"Class distribution before:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} samples")

    if args.smote_strategy == 'auto':
        smote = SMOTE(random_state=42, k_neighbors=5)
    else:
        # Parse strategy like "1:2000,6:2000"
        strategy_dict = {}
        for item in args.smote_strategy.split(','):
            cls, cnt = item.split(':')
            strategy_dict[int(cls)] = int(cnt)
        smote = SMOTE(sampling_strategy=strategy_dict, random_state=42, k_neighbors=5)

    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"\nAfter SMOTE: {len(X_train_resampled)} samples")
    print(f"Class distribution after:")
    unique, counts = np.unique(y_train_resampled, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} samples")

    # Normalize using training statistics (from resampled data)
    traj_mean = X_train_resampled.mean(axis=0)
    traj_std = X_train_resampled.std(axis=0) + 1e-8

    X_train_norm = (X_train_resampled - traj_mean) / traj_std
    X_test_norm = (X_test - traj_mean) / traj_std

    # Reshape for BiLSTM: (N, 512, 27)
    # Each sample is a sequence of 512 timesteps with 27 features
    seq_len = 512
    num_features = 27

    # Check if data needs reshaping
    if X_train_norm.shape[1] != seq_len * num_features:
        raise ValueError(f"Expected {seq_len * num_features} features, got {X_train_norm.shape[1]}")

    X_train_seq = X_train_norm.reshape(-1, seq_len, num_features)
    X_test_seq = X_test_norm.reshape(-1, seq_len, num_features)

    print(f"\nReshaped data: train {X_train_seq.shape}, test {X_test_seq.shape}")

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_seq)
    y_train_tensor = torch.LongTensor(y_train_resampled)
    X_test_tensor = torch.FloatTensor(X_test_seq)
    y_test_tensor = torch.LongTensor(y_test)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = AVmodel(
        mode='trajectory_only',
        num_classes=11,
        traj_arch='bilstm',
        adapter_dim=8,
        num_latent=4
    ).to(device)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n\tModel loaded (mode=trajectory_only)")
    print(f"\tTrainable params = {trainable_params}")

    # Loss: standard CrossEntropy (no class weights, SMOTE already balanced)
    loss_fn = nn.CrossEntropyLoss()
    print(f"\tLoss: CrossEntropy (no class weights, SMOTE balanced)")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Training loop
    print(f"\n\tStarted training\n")
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    best_model_state = None

    for epoch in range(args.num_epochs):
        # Train
        model.train()
        epoch_loss = []
        correct = 0
        total = 0

        for batch_idx, (traj, labels) in enumerate(trainloader):
            traj = traj.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # Create dummy images (not used in trajectory_only mode)
            dummy_imgs = torch.zeros(traj.size(0), 8, 3, 224, 224).to(device)
            preds = model(traj, dummy_imgs)
            loss = loss_fn(preds, labels)

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            correct += (torch.argmax(preds, dim=1) == labels).sum().item()
            total += len(labels)

        train_loss = np.mean(epoch_loss)
        train_acc = correct / total * 100

        # Validate
        model.eval()
        val_loss = []
        correct = 0
        total = 0

        with torch.no_grad():
            for traj, labels in testloader:
                traj = traj.to(device)
                labels = labels.to(device)
                dummy_imgs = torch.zeros(traj.size(0), 8, 3, 224, 224).to(device)
                preds = model(traj, dummy_imgs)
                loss = loss_fn(preds, labels)

                val_loss.append(loss.item())
                correct += (torch.argmax(preds, dim=1) == labels).sum().item()
                total += len(labels)

        val_loss = np.mean(val_loss)
        val_acc = correct / total * 100

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1:3d}/{args.num_epochs}"
              f"  train loss {train_loss:.4f}  acc {train_acc:.2f}%"
              f"  val loss {val_loss:.4f}  val acc {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

    # Load best model and evaluate
    model.load_state_dict(best_model_state)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for traj, labels in testloader:
            traj = traj.to(device)
            dummy_imgs = torch.zeros(traj.size(0), 8, 3, 224, 224).to(device)
            preds = model(traj, dummy_imgs)
            all_preds.extend(torch.argmax(preds, dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

    # Compute metrics
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)

    # Save results
    results = {
        'experiment': 'trajectory_only_smote',
        'smote_strategy': args.smote_strategy,
        'train_samples_before_smote': len(X_train),
        'train_samples_after_smote': len(X_train_resampled),
        'test_samples': len(X_test),
        'best_val_acc': best_val_acc,
        'per_class_metrics': {
            f'class_{i}': {
                'precision': report[str(i)]['precision'] * 100,
                'recall': report[str(i)]['recall'] * 100,
                'f1-score': report[str(i)]['f1-score'] * 100,
                'support': report[str(i)]['support']
            }
            for i in range(11)
        },
        'macro_avg': {
            'precision': report['macro avg']['precision'] * 100,
            'recall': report['macro avg']['recall'] * 100,
            'f1-score': report['macro avg']['f1-score'] * 100
        },
        'history': history
    }

    # Save to file
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, 'results_trajectory_smote.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {output_file}")
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    print(f"Macro F1: {report['macro avg']['f1-score'] * 100:.2f}%")

    # Print per-class metrics
    print("\nPer-class metrics:")
    for i in range(11):
        metrics = results['per_class_metrics'][f'class_{i}']
        print(f"  Class {i:2d}: P={metrics['precision']:.2f}%  R={metrics['recall']:.2f}%  F1={metrics['f1-score']:.2f}%  (n={int(metrics['support'])})")


if __name__ == '__main__':
    main()
