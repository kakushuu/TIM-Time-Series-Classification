#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main training script for BiLSTM trajectory classification
Experiments:
1. trajectory_only: BiLSTM with trajectory features only
2. multimodal: BiLSTM with trajectory + image features
"""

import os
import torch
import argparse
import opt
from models.lstm import AttBiLSTM, AttBiLSTM_Multimodal
from trainer import Trainer
import warnings
warnings.filterwarnings("ignore")


def train_trajectory_only():
    """Experiment 1: BiLSTM with trajectory features only"""
    print("\n" + "="*70)
    print("Experiment 1: BiLSTM Trajectory Only")
    print("="*70)

    # Update config for trajectory-only experiment
    opt.mode = "trajectory_only"
    opt.filename = "bilstm_trajectory_only"
    opt.NAME = "BiLSTM_Trajectory_Only"

    # Create model
    model = AttBiLSTM(
        n_classes=opt.n_classes,
        emb_size=opt.emb_size,
        rnn_size=opt.rnn_size,
        rnn_layers=opt.rnn_layers,
        dropout=opt.dropout
    )

    print(f"\nModel: {opt.NAME}")
    print(f"  - Input features: {opt.emb_size}")
    print(f"  - RNN size: {opt.rnn_size}")
    print(f"  - RNN layers: {opt.rnn_layers}")
    print(f"  - Classes: {opt.n_classes}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer = Trainer(model, mode='trajectory_only')
    trainer.start_train()

    return trainer.best_acc


def train_multimodal():
    """Experiment 2: BiLSTM with trajectory + image features"""
    print("\n" + "="*70)
    print("Experiment 2: BiLSTM Multimodal (Trajectory + Image)")
    print("="*70)

    # Update config for multimodal experiment
    opt.mode = "multimodal"
    opt.filename = "bilstm_multimodal"
    opt.NAME = "BiLSTM_Multimodal"

    # Image feature size (ViT-B16 output)
    img_feat_size = 768

    # Create model
    model = AttBiLSTM_Multimodal(
        n_classes=opt.n_classes,
        traj_emb_size=opt.emb_size,
        img_feat_size=img_feat_size,
        rnn_size=opt.rnn_size,
        rnn_layers=opt.rnn_layers,
        dropout=opt.dropout
    )

    print(f"\nModel: {opt.NAME}")
    print(f"  - Trajectory features: {opt.emb_size}")
    print(f"  - Image features: {img_feat_size}")
    print(f"  - RNN size: {opt.rnn_size}")
    print(f"  - RNN layers: {opt.rnn_layers}")
    print(f"  - Classes: {opt.n_classes}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer = Trainer(model, mode='multimodal')
    trainer.start_train()

    return trainer.best_acc


def main():
    parser = argparse.ArgumentParser(description="BiLSTM Trajectory Classification")
    parser.add_argument(
        '--mode',
        choices=['trajectory_only', 'multimodal', 'all'],
        default='all',
        help='Which experiment to run'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='/home/research/Agri-MBT/data/aligned_output/aligned_data.csv',
        help='Path to aligned data CSV'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=3e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )

    args = parser.parse_args()

    # Update config
    opt.data_dir = args.data
    opt.epochs = args.epochs
    opt.batch_size = args.batch_size
    opt.LEARNING_RATE = args.lr
    opt.device = torch.device(args.device)

    print("\n" + "="*70)
    print("BiLSTM Trajectory Classification Experiments")
    print("="*70)
    print(f"Data: {opt.data_dir}")
    print(f"Epochs: {opt.epochs}")
    print(f"Batch size: {opt.batch_size}")
    print(f"Learning rate: {opt.LEARNING_RATE}")
    print(f"Device: {opt.device}")
    print(f"Classes: {opt.n_classes}")
    print("="*70)

    # Run experiments
    results = {}

    if args.mode in ['trajectory_only', 'all']:
        best_acc = train_trajectory_only()
        results['trajectory_only'] = best_acc

    if args.mode in ['multimodal', 'all']:
        best_acc = train_multimodal()
        results['multimodal'] = best_acc

    # Print summary
    print("\n" + "="*70)
    print("Experiment Summary")
    print("="*70)
    for exp_name, acc in results.items():
        print(f"{exp_name:20s}: Best Val Acc = {acc:.2f}%")
    print("="*70 + "\n")

    return results


if __name__ == "__main__":
    main()
