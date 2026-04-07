---
title: Trajectory Model Innovation Experiments - Failed Attempts
date: 2026-03-17
tags: [ml, trajectory, classification, failed-experiments, ablation]
---

# Trajectory Model Innovation Experiments

**Date**: 2026-03-17  
**Objective**: Improve trajectory_only baseline (80.37%) and multimodal baseline (94.18%)  
**Result**: All three innovations FAILED to improve over baselines

## Executive Summary

This experiment series tested three novel architectures to improve GPS trajectory classification:
1. **PatchTST** (Transformer-based, 29M params)
2. **Multimodal Contrastive Regularization** (InfoNCE loss)
3. **Hierarchical BiLSTM** (dual-scale temporal encoder)
4. **Trajectory-as-Image CNN** (ResNet-18 on trajectory "images")

**Key Finding**: None of the innovations improved performance. The baseline BiLSTM + weighted CE remains optimal.

## Experiment Results

### Trajectory_only Experiments (Baseline: 80.37%)

| Experiment | Val Acc | Δ vs Baseline | Status | Issue |
|-----------|---------|---------------|--------|-------|
| ✅ **BiLSTM + WeightedCE** | **80.37%** | — | BASELINE | — |
| ❌ PatchTST + CE (lr=3e-4) | 42.47% | -37.90% | FAILED | Class collapse (class_7 only) |
| ❌ PatchTST + CE (lr=1e-3) | 42.22% | -38.15% | FAILED | Class collapse (class_7 only) |
| ❌ PatchTST + Focal γ=2.0 | 7.26% | -73.11% | FAILED | Training collapse |
| ❌ PatchTST + Focal γ=0.5 | ~10% | -70.37% | FAILED | Training collapse |
| ❌ Hierarchical BiLSTM | 75.96% | -4.41% | FAILED | Oscillating (75%→58%→76%) |
| ❌ Trajectory-as-Image CNN | 40.66% | -39.71% | FAILED | Catastrophic collapse |

### Multimodal Experiments (Baseline: 94.18%)

| Experiment | Val Acc | Δ vs Baseline | Status | Issue |
|-----------|---------|---------------|--------|-------|
| ✅ **Multimodal (ViT+BiLSTM)** | **94.18%** | — | BASELINE | — |
| ❌ Multimodal + Contrastive | 93.11% | -1.07% | FAILED | Overfitting (peak 93.97% → 93.11%) |

## Root Cause Analysis

### PatchTST Failures
- **Problem**: 29M parameter Transformer consistently collapsed to predicting class_7 (majority class)
- **Attempted fixes**: 
  - Higher learning rate (3e-4 → 1e-3): No improvement
  - Focal loss (γ=2.0, γ=0.5): Worse performance
- **Diagnosis**: Patch-based attention not suitable for GPS trajectory sequences; BiLSTM's sequential inductive bias is essential

### Multimodal Contrastive Failure
- **Problem**: InfoNCE contrastive loss caused overfitting
- **Training dynamics**:
  - Peak at Epoch 4: 93.97% (0.21% below baseline)
  - Degraded to Epoch 9: 93.11% (1.07% below baseline)
  - Val loss spiked to 0.7875 (highest since Epoch 1)
  - Train acc: 97.25% >> Val acc: 93.11%
- **Diagnosis**: Cross-modal contrastive alignment conflicts with MBT's attention bottleneck; vision already dominates trajectory

### Hierarchical BiLSTM Failure
- **Problem**: Severe training instability (oscillating val acc)
- **Training dynamics**:
  - Epoch 1: 75.23%
  - Epoch 2: 58.82% (massive drop -16.41%)
  - Epoch 3: 75.96% (recovered but still below baseline)
- **Diagnosis**: Dual-scale BiLSTM (512 frames + 51 downsampled) creates conflicting gradients; 11.34M params too large

### Trajectory-as-Image CNN Failure
- **Problem**: Catastrophic collapse to class_7 prediction
- **Training dynamics**:
  - Epoch 1: 42.47%
  - Epoch 2: 40.66% (degrading)
- **Diagnosis**: Resizing (27, 512) → (27, 224, 224) destroys temporal structure; CNN spatial inductive bias inappropriate for time series

## Key Insights

1. **BiLSTM's sequential bias is essential**: PatchTST (Transformer) and CNN architectures consistently failed
2. **Class imbalance is the core challenge**: Weighted CE is necessary but not sufficient
3. **Vision dominates trajectory**: Multimodal model works (94.18%) because visual features carry most information
4. **Trajectory encoding is harder than expected**: Even simple improvements (hierarchical, CNN) degraded performance

## Recommendations

1. **Keep baseline**: BiLSTM + weighted CE (80.37%) remains optimal for trajectory_only
2. **Focus on data**: Feature engineering or data augmentation may help more than architecture changes
3. **Accept vision dominance**: Multimodal model (94.18%) is already strong; trajectory improvements are marginal

## Files Included

- Training logs for all experiments
- Result JSON files (where available)
- Model checkpoints (excluded - too large)

## Next Steps

- Try different trajectory features (beyond 27-dim engineered features)
- Experiment with data augmentation (time warping, jittering)
- Consider simpler models (XGBoost, Random Forest) as baselines
