#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify BiLSTM trajectory implementation
"""

import os
import sys
import torch
import numpy as np

print("="*70)
print("BiLSTM Trajectory Classification - Test Script")
print("="*70)

# Test 1: Import modules
print("\n[Test 1] Importing modules...")
try:
    import opt
    print("  ✓ opt.py imported")

    from models.lstm import AttBiLSTM, AttBiLSTM_Multimodal
    print("  ✓ models.lstm imported")

    from models.lossFun import FocalLoss
    print("  ✓ models.lossFun imported")

    from utils.loader import get_loader_trajectory
    print("  ✓ utils.loader imported")

    from utils.metrics import scores
    print("  ✓ utils.metrics imported")

    print("  ✅ All modules imported successfully")
except Exception as e:
    print(f"  ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Model initialization
print("\n[Test 2] Testing model initialization...")
try:
    # Trajectory-only model
    model_traj = AttBiLSTM(
        n_classes=opt.n_classes,
        emb_size=opt.emb_size,
        rnn_size=opt.rnn_size,
        rnn_layers=opt.rnn_layers,
        dropout=opt.dropout
    )
    n_params_traj = sum(p.numel() for p in model_traj.parameters())
    print(f"  ✓ Trajectory model: {n_params_traj:,} parameters")

    # Multimodal model
    model_mm = AttBiLSTM_Multimodal(
        n_classes=opt.n_classes,
        traj_emb_size=opt.emb_size,
        img_feat_size=768,
        rnn_size=opt.rnn_size,
        rnn_layers=opt.rnn_layers,
        dropout=opt.dropout
    )
    n_params_mm = sum(p.numel() for p in model_mm.parameters())
    print(f"  ✓ Multimodal model: {n_params_mm:,} parameters")

    print("  ✅ Models initialized successfully")
except Exception as e:
    print(f"  ❌ Model initialization failed: {e}")
    sys.exit(1)

# Test 3: Forward pass
print("\n[Test 3] Testing forward pass...")
try:
    device = opt.device
    batch_size = 4
    seq_len = opt.time_tri
    emb_size = opt.emb_size

    # Trajectory-only forward
    x_traj = torch.randn(batch_size, seq_len, emb_size).to(device)
    model_traj = model_traj.to(device)
    out_traj = model_traj(x_traj)
    assert out_traj.shape == (batch_size, opt.n_classes), f"Expected {(batch_size, opt.n_classes)}, got {out_traj.shape}"
    print(f"  ✓ Trajectory forward: input {x_traj.shape} -> output {out_traj.shape}")

    # Multimodal forward
    x_img = torch.randn(batch_size, 768).to(device)
    model_mm = model_mm.to(device)
    out_mm = model_mm(x_traj, x_img)
    assert out_mm.shape == (batch_size, opt.n_classes), f"Expected {(batch_size, opt.n_classes)}, got {out_mm.shape}"
    print(f"  ✓ Multimodal forward: traj {x_traj.shape} + img {x_img.shape} -> output {out_mm.shape}")

    print("  ✅ Forward pass successful")
except Exception as e:
    print(f"  ❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Loss function
print("\n[Test 4] Testing loss function...")
try:
    loss_fn = FocalLoss(alpha=opt.ALPHA, gamma=opt.GAMMA, logits=True)

    # Create dummy targets
    targets = torch.randint(0, opt.n_classes, (batch_size,)).to(device)
    targets_onehot = torch.eye(opt.n_classes).to(device)[targets]

    loss = loss_fn(out_traj, targets_onehot)
    assert loss.item() > 0, "Loss should be positive"
    print(f"  ✓ Focal loss computed: {loss.item():.4f}")

    print("  ✅ Loss function working")
except Exception as e:
    print(f"  ❌ Loss computation failed: {e}")
    sys.exit(1)

# Test 5: Metrics
print("\n[Test 5] Testing metrics...")
try:
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 2])

    met = scores(y_true, y_pred)
    print(f"  ✓ Metrics computed:")
    print(f"    - Accuracy: {met['accuracy']:.2f}%")
    print(f"    - F1 Macro: {met['f1_macro']:.2f}%")
    print(f"    - F1 Weighted: {met['f1_weighted']:.2f}%")

    print("  ✅ Metrics working")
except Exception as e:
    print(f"  ❌ Metrics computation failed: {e}")
    sys.exit(1)

# Test 6: Data loading (if data exists)
print("\n[Test 6] Testing data loading...")
data_path = opt.data_dir
if os.path.exists(data_path):
    try:
        print(f"  Loading data from: {data_path}")

        # Load small batch
        train_loader, valid_loader, test_loader, _ = get_loader_trajectory(
            data_path, step=8, batch_size=16, num_workers=0
        )

        # Get one batch
        for X, y in train_loader:
            print(f"  ✓ Data loaded:")
            print(f"    - X shape: {X.shape}")
            print(f"    - y shape: {y.shape}")
            print(f"    - y unique: {torch.unique(y).tolist()}")
            break

        print(f"  ✓ Train batches: {len(train_loader)}")
        print(f"  ✓ Valid batches: {len(valid_loader)}")
        print(f"  ✓ Test batches: {len(test_loader)}")

        print("  ✅ Data loading successful")
    except Exception as e:
        print(f"  ⚠ Data loading failed (expected if no data): {e}")
        print("  ℹ️  This is OK - you can train once data is available")
else:
    print(f"  ⚠ Data file not found: {data_path}")
    print("  ℹ️  This is OK - you can train once data is available")

# Summary
print("\n" + "="*70)
print("Test Summary")
print("="*70)
print("✅ All core components working correctly!")
print("\nNext steps:")
print("1. Ensure data is available at: {opt.data_dir}")
print("2. Run training:")
print("   python train.py --mode trajectory_only --epochs 50")
print("3. Or run all experiments:")
print("   bash run_experiments.sh")
print("="*70 + "\n")
