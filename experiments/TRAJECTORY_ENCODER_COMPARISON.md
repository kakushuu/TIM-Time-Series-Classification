# Trajectory-Only Experiments Summary

**Date**: 2026-03-17
**Goal**: Find alternative trajectory encoder to match or exceed BiLSTM's 80.37% accuracy

---

## Experiments Overview

| Model | Parameters | Epochs | Best Val Acc | Status | Notes |
|-------|-----------|--------|--------------|--------|-------|
| **BiLSTM + Attention** | 8.6M | 15 | **80.37%** | ✅ **Baseline** | Attention pooling over all hidden states (Bug A1 fixed) |
| Trajectory-as-Image (ResNet-18) | 8.6M | 15 | **38.95%** | ❌ **Failed** | Collapsed to class_7 only (dominant class) |
| PatchTST (Transformer) | 29.2M | 6/20 | **38.95%** | ❌ **Failed** | Stuck at 38.95% (class_7 frequency) |
| **Hierarchical BiLSTM** | 11.3M | - | - | 🔄 **Training** | Dual-scale: short-term (512 frames) + long-term (51 frames) |

---

## Failed Approaches Analysis

### 1. Trajectory-as-Image (ResNet-18) — 38.95% ❌

**Architecture:**
```
(512, 27) → reshape → (27, 512, 1) → interpolate → (27, 224, 224)
→ ResNet-18 (ImageNet pretrained) → 512-dim → classifier
```

**Per-Class Metrics:**
- **Classes 0-6, 8-10**: 0% recall (complete failure)
- **Class 7 (dominant)**: 100% recall, 38.95% precision (predicted EVERYTHING as class_7)
- **Macro F1**: 5.10% (vs BiLSTM's 49.70%)

**Why it failed:**
1. **Information loss**: (512, 27) → (224, 224) interpolation destroys temporal structure
2. **Spatial vs Temporal**: ResNet looks for spatial patterns, but the signal is temporal
3. **Pretrained mismatch**: ImageNet features don't transfer to trajectory heatmaps
4. **Training time**: ~2 hours (slow due to interpolation)

---

### 2. PatchTST (Transformer) — 38.95% ❌

**Architecture:**
```
(512, 27) → patch size 16 → 32 patches × (16×27=432) dims
→ Linear projection → 32 tokens × 768 dims
→ 4-layer Transformer encoder (8 heads)
→ Mean pooling → 768-dim → classifier
```

**Training Progress:**
```
Epoch   1/20  val acc 18.89%
Epoch   2/20  val acc 18.89%
Epoch   3/20  val acc 38.95%  ← jumped to class_7 frequency
Epoch   4-6/20  val acc 38.95%  ← stuck
```

**Why it failed:**
1. **Patch size mismatch**: 16-frame patches may be too coarse for 512-frame sequences
2. **Transformer overfitting**: 29M parameters overfit to dominant class
3. **Positional encoding**: May not capture trajectory dynamics well
4. **Lack of attention pooling**: Mean pooling loses fine-grained information

---

## Successful Approach: BiLSTM + Attention

**Architecture:**
```
(512, 27) → 2-layer BiLSTM (384 hidden × 2 directions = 768)
→ Attention pooling over all 512 hidden states
→ 768-dim → 2-layer MLP (768→512→256) → classifier
```

**Key Success Factors:**
1. ✅ **Attention pooling**: Learns which timesteps are important
2. ✅ **Class-weighted loss**: Handles 31.85:1 class imbalance
3. ✅ **Bidirectional context**: Both past and future context
4. ✅ **Sequential inductive bias**: LSTM designed for time series

**Per-Class Performance:**
- **Class 7**: 97.46% recall (dominant, but NOT exclusive)
- **Class 10**: 94.01% recall, 98.76% precision (excellent)
- **Class 3**: 64.08% recall, 78.95% precision (good)
- **Class 1**: 0% recall (failed on rarest class with 393 samples)

---

## Current Experiment: Hierarchical BiLSTM

**Hypothesis:** Combine short-term fast motion patterns + long-term global trajectory shape

**Architecture:**
```
Short-term branch:
  (512, 27) → 2-layer BiLSTM → attention pooling → 768-dim

Long-term branch:
  (51, 27) [every 10th frame] → 2-layer BiLSTM → attention pooling → 768-dim

Fusion:
  concat(768, 768) → Linear(1536 → 768) → 768-dim → classifier
```

**Parameters**: 11.3M (2× BiLSTM parameters)

**Expected benefits:**
- ✅ Short-term: Captures rapid motion changes (speed, acceleration spikes)
- ✅ Long-term: Captures global trajectory shape (field boundaries, loops)
- ✅ Multi-scale: Like hierarchical CNN, but for sequences

**Training started**: 2026-03-17 22:20

---

## Next Steps

1. ✅ **Wait for Hierarchical BiLSTM results** (ETA: ~1.5 hours)
2. 📊 **Compare all approaches** on:
   - Overall accuracy
   - Macro F1-score (class balance)
   - Per-class recall (especially rare classes 1, 6)
   - Training efficiency
3. 📝 **Generate comprehensive report** with:
   - Architecture diagrams
   - Per-class precision/recall/F1 tables
   - Training curves
   - Best practices recommendations
4. 🚀 **Optional experiments**:
   - Try **Focal Loss** instead of weighted CE
   - **SMOTE oversampling** for rare classes
   - **Ensemble** BiLSTM + Hierarchical BiLSTM

---

## Key Insights

| Factor | Success (BiLSTM) | Failure (ResNet/PatchTST) |
|--------|------------------|---------------------------|
| **Temporal modeling** | ✅ LSTM (designed for sequences) | ❌ Spatial architectures |
| **Attention mechanism** | ✅ Attention pooling (learned focus) | ❌ Mean/max pooling |
| **Class imbalance** | ✅ Weighted CE + attention | ❌ Weighted CE alone insufficient |
| **Information preservation** | ✅ No interpolation | ❌ (512,27)→(224,224) loss |
| **Inductive bias** | ✅ Sequential | ❌ Spatial (ResNet) / Local patches (Transformer) |

**Conclusion**: For GPS trajectory classification, **temporal architectures with learned attention** significantly outperform spatial/patch-based approaches.
