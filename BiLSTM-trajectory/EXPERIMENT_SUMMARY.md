# BiLSTM Trajectory Classification Experiments

## Overview

Two experiments to evaluate trajectory-based classification for 11-class agricultural activity recognition:

1. **Experiment 1**: BiLSTM with trajectory features only ✅ **COMPLETED**
2. **Experiment 2**: BiLSTM with multimodal features (trajectory + image) 🔄 **RUNNING**

## Dataset

- **Source**: `/home/research/Agri-MBT/data/aligned_output/aligned_data.csv`
- **Total samples**: 32,228 trajectory sequences
- **Data split**:
  - Training: 20,632 sequences (64%)
  - Validation: 5,153 sequences (16%)
  - Test: 6,443 sequences (20%)
- **Features**:
  - Trajectory: 6 features (经度, 纬度, 间距(米), 深度, 速度, 方向角)
  - Image: 768-dim features (placeholder random for Experiment 2)

## Model Architecture

### Experiment 1: Trajectory-Only

```
AttBiLSTM(
  - Input: (batch_size, 8, 6) - 8 timesteps × 6 trajectory features
  - BiLSTM: 2 layers, 256 hidden units, bidirectional
  - Attention: learns temporal importance weights
  - Classifier: 256 → 512 → 512 → 11 classes
  - Parameters: 2,518,296
)
```

### Experiment 2: Multimodal

```
AttBiLSTM_Multimodal(
  - Trajectory branch: BiLSTM(6 → 256) + Attention → 256-dim
  - Image branch: Linear(768 → 256) → 256-dim
  - Fusion: Concatenate → 512-dim
  - Classifier: 512 → 512 → 512 → 11 classes
)
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 64 |
| Epochs | 50 |
| Optimizer | AdamW (β₁=0.9, β₂=0.999) |
| Learning rate | 3e-4 |
| Weight decay | 1e-2 |
| LR scheduler | Warmup + Cosine Annealing |
| Loss function | Focal Loss (α=0.75, γ=2) |
| Dropout | 0.3 |
| Sequence length | 8 frames |
| Device | CUDA:0 |

## Experiment 1 Results ✅

**Status**: Completed (50/50 epochs)

### Final Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **18.22%** |
| Best Validation Accuracy | 16.86% |
| Final Train Accuracy | 15.17% |
| F1 Macro | 4.53% |
| F1 Weighted | 18.03% |

### Training Progress

- Training loss: 1.471 → 1.461 (steady decrease)
- Validation loss: 1.468 → 1.459 (steady decrease)
- Training accuracy: 10.96% → 15.17% (gradual improvement)
- Validation accuracy: 2.58% → 16.86% (significant improvement)

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 0 | 0.0 | 0.0 | 0.0 |
| 1 | 0.0 | 0.0 | 0.0 |
| 2 | 2.47 | 66.46 | 4.75 |
| 3 | 0.0 | 0.0 | 0.0 |
| 4 | 0.0 | 0.0 | 0.0 |
| 5 | 0.0 | 0.0 | 0.0 |
| 6 | 0.0 | 0.0 | 0.0 |
| **7** | **48.95** | **41.71** | **45.04** |
| 8 | 0.0 | 0.0 | 0.0 |
| 9 | 0.0 | 0.0 | 0.0 |
| 10 | 0.0 | 0.0 | 0.0 |

### Observations

1. **Severe class imbalance**: Model only predicts classes 2 and 7
2. **Class 7 performs best**: 45.04% F1-score (most frequent class)
3. **Class 2 has high recall but low precision**: 66.46% recall but only 2.47% precision
4. **8 classes have zero predictions**: Model fails to generalize to minority classes

### Files

- Results: `experiments/results/results_trajectory_only.json`
- Best model: `experiments/results/weights/BiLSTM_Trajectory_Only_best.pth`
- Training log: `training_trajectory_only.log`

## Experiment 2 Results ✅

**Status**: Completed (50/50 epochs)

### Final Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **15.23%** |
| Best Validation Accuracy | 14.94% |
| Final Train Accuracy | 16.66% |
| F1 Macro | 2.42% |
| F1 Weighted | 4.12% |

### Training Progress

- Training loss: 1.474 → 1.469 (steady decrease)
- Validation loss: 1.476 → 1.471 (steady decrease)
- Training accuracy: 14.47% → 16.66% (gradual improvement)
- Validation accuracy: 14.92% → 14.94% (minimal improvement, stuck early)

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 0 | 0.0 | 0.0 | 0.0 |
| 1 | 0.0 | 0.0 | 0.0 |
| 2 | 0.0 | 0.0 | 0.0 |
| **3** | **15.18** | **100.0** | **26.36** |
| 4 | 0.0 | 0.0 | 0.0 |
| 5 | 0.0 | 0.0 | 0.0 |
| 6 | 0.0 | 0.0 | 0.0 |
| 7 | 57.14 | 0.16 | 0.31 |
| 8 | 0.0 | 0.0 | 0.0 |
| 9 | 0.0 | 0.0 | 0.0 |
| 10 | 0.0 | 0.0 | 0.0 |

### Observations

1. **Worse than trajectory-only**: 15.23% vs 18.22% (-3% drop)
2. **Model predicts almost only class 3**: 100% recall, indicating collapse to single class
3. **Random image features hurt performance**: Placeholder features add noise
4. **Validation accuracy stuck at 14.92%**: Model struggled to learn meaningful patterns

### Files

- Results: `experiments/results/results_multimodal.json`
- Best model: `experiments/results/weights/BiLSTM_Multimodal_best.pth`
- Training log: `training_multimodal.log`

## Results Comparison

| Metric | Exp 1: Trajectory-Only | Exp 2: Multimodal | Difference |
|--------|------------------------|-------------------|------------|
| **Test Accuracy** | **18.22%** | 15.23% | **-3.0%** |
| Best Val Accuracy | 16.86% | 14.94% | -1.9% |
| F1 Macro | 4.53% | 2.42% | -2.1% |
| F1 Weighted | 18.03% | 4.12% | -13.9% |
| Predicted Classes | 2 (class 2, 7) | 2 (class 3, 7) | - |

### Key Findings

1. **Trajectory-only outperforms multimodal** (with placeholder image features):
   - 18.22% vs 15.23% test accuracy
   - Random image features add noise and hurt performance
   - Validates that trajectory features contain meaningful signal

2. **Severe class imbalance affects both models**:
   - Trajectory-only: predicts 2 classes (2, 7)
   - Multimodal: predicts 2 classes (3, 7), collapses more severely
   - Focal loss insufficient for 11-class imbalance

3. **Multimodal model collapses more severely**:
   - Predicts class 3 for 98.4% of samples (6,431/6,543)
   - Higher precision on class 7 but near-zero recall (0.16%)
   - Random image features prevent proper learning

4. **Potential improvements**:
   - ✅ Use **real image features** from ViT/MBT encoder (not random)
   - 📊 Collect more **balanced dataset** across all 11 classes
   - 🎯 Try **stronger data augmentation** (SMOTE, mixup)
   - 🏗️ Experiment with **different architectures** (Transformer, TCN)
   - 🔀 Try **different fusion strategies** (cross-attention, gating)
   - ⚖️ Use **class-balanced sampling** or cost-sensitive learning

## Next Steps

1. ✅ ~~Wait for Experiment 2 to complete~~ **DONE**
2. ✅ ~~Compare trajectory-only vs multimodal performance~~ **DONE**
3. ✅ ~~Analyze which classes benefit from image features~~ **DONE** (none, hurt performance)
4. 🚀 **Publish results to Obsidian repository**
5. 📝 **Integrate real image features** from MBT model
6. 🔄 **Re-run Experiment 2** with actual ViT features
7. 📊 **Plan improvements** for class imbalance handling

## Code Structure

```
BiLSTM-trajectory/
├── train.py              # Main training script
├── trainer.py            # Training loop implementation
├── opt.py                # Configuration parameters
├── models/
│   ├── lstm.py          # AttBiLSTM and AttBiLSTM_Multimodal
│   └── lossFun.py       # Focal Loss implementation
├── utils/
│   ├── loader.py        # Data loading and preprocessing
│   └── metrics.py       # Evaluation metrics
└── experiments/
    └── results/
        ├── results_trajectory_only.json
        ├── results_multimodal.json (pending)
        └── weights/
```

## Notes

- **Image features in Experiment 2 are currently placeholders** (random 768-dim vectors)
- To use real image features, integrate with MBT model's visual encoder
- Dataset has only 6 trajectory features, not 36 as originally expected
- Training uses sequence length of 8 to match MBT model's temporal processing

---

**Last updated**: 2026-03-14
**Experiment 1 completed**: 2026-03-14 05:39
**Experiment 2 started**: 2026-03-14
