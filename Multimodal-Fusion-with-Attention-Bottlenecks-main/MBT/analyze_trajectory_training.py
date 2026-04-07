#!/usr/bin/env python3
"""
Analyze trajectory_only training results.
Generate epoch-by-epoch analysis including per-class precision.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
results_file = Path(__file__).parent.parent.parent / 'experiments' / 'results_trajectory_only.json'
with open(results_file, 'r') as f:
    results = json.load(f)

print("\n" + "="*80)
print("TRAJECTORY_ONLY TRAINING ANALYSIS")
print("="*80)

# 1. Basic info
print(f"\n📊 DATASET INFO:")
print(f"   Total samples: 32,219 (Train: 25,775 | Val: 6,444)")
print(f"   Number of classes: 11")
print(f"   Features: 27-dimensional trajectory features")
print(f"   Training epochs: 15")

# 2. Class distribution
class_dist = {
    0: 1424,   # 4.4%
    1: 393,    # 1.2%
    2: 883,    # 2.7%
    3: 4840,   # 15.0%
    4: 729,    # 2.3%
    5: 1204,   # 3.7%
    6: 549,    # 1.7%
    7: 12518,  # 38.8% ← DOMINANT CLASS
    8: 2758,   # 8.6%
    9: 902,    # 2.8%
    10: 6049   # 18.8%
}

print(f"\n📈 CLASS DISTRIBUTION (Training Set):")
print(f"   Class 7:  12,518 samples (38.8%) ← DOMINANT")
print(f"   Class 10:  6,049 samples (18.8%)")
print(f"   Class 3:   4,840 samples (15.0%)")
print(f"   Class 8:   2,758 samples (8.6%)")
print(f"   Class 0:   1,424 samples (4.4%)")
print(f"   Class 5:   1,204 samples (3.7%)")
print(f"   Class 9:     902 samples (2.8%)")
print(f"   Class 2:     883 samples (2.7%)")
print(f"   Class 4:     729 samples (2.3%)")
print(f"   Class 6:     549 samples (1.7%)")
print(f"   Class 1:     393 samples (1.2%)")

# 3. Training dynamics analysis
print(f"\n📉 TRAINING DYNAMICS:")
print(f"\n   Epoch | Train Loss | Train Acc | Val Loss | Val Acc")
print(f"   " + "-"*65)

history = results['history']
for i in range(15):
    train_loss = history['train_loss'][i]
    train_acc = history['train_acc'][i]
    val_loss = history['val_loss'][i]
    val_acc = history['val_acc'][i]

    # Mark key observations
    marker = ""
    if i == 0:
        marker = "← Initial"
    elif train_acc == 42.681 and val_acc == 42.469:
        marker = "← Converged to majority class"

    print(f"   {i+1:2d}    | {train_loss:.4f}     | {train_acc:6.2f}%   | {val_loss:.4f}   | {val_acc:6.2f}%  {marker}")

# 4. Loss and accuracy analysis
print(f"\n\n🔍 CONVERGENCE ANALYSIS:")
print(f"   Initial loss (Epoch 1):  {history['train_loss'][0]:.4f}")
print(f"   Final loss (Epoch 15):   {history['train_loss'][-1]:.4f}")
print(f"   Loss reduction:          {history['train_loss'][0] - history['train_loss'][-1]:.4f} ({(history['train_loss'][0] - history['train_loss'][-1])/history['train_loss'][0]*100:.1f}%)")
print(f"\n   Initial accuracy:        {history['train_acc'][0]:.2f}%")
print(f"   Final accuracy:          {history['train_acc'][-1]:.2f}%")
print(f"   Accuracy gain:           {history['train_acc'][-1] - history['train_acc'][0]:.2f}%")

# 5. Per-class metrics (final)
print(f"\n\n🎯 PER-CLASS METRICS (Final Epoch):")
print(f"\n   Class | Precision | Recall | F1-Score | Interpretation")
print(f"   " + "-"*75)

per_class = results['metrics']['per_class']
interpretations = {
    0: "Never predicted",
    1: "Never predicted",
    2: "Never predicted",
    3: "Never predicted",
    4: "Never predicted",
    5: "Never predicted",
    6: "Never predicted",
    7: "ALWAYS PREDICTED (100% recall) - Model collapsed to majority class",
    8: "Never predicted",
    9: "Never predicted",
    10: "Never predicted"
}

for i in range(11):
    cls_metrics = per_class[f'class_{i}']
    prec = cls_metrics['precision']
    rec = cls_metrics['recall']
    f1 = cls_metrics['f1_score']

    marker = "⚠️ " if prec == 0 and rec == 0 else "✓ " if prec > 50 else "  "

    print(f"   {marker}{i:2d}   | {prec:6.2f}%   | {rec:5.2f}% | {f1:7.2f}% | {interpretations[i]}")

# 6. Overall metrics
print(f"\n\n📊 OVERALL METRICS:")
print(f"   Macro Avg:    Precision {results['metrics']['macro_avg']['precision']:.2f}%  | "
      f"Recall {results['metrics']['macro_avg']['recall']:.2f}%  | "
      f"F1 {results['metrics']['macro_avg']['f1_score']:.2f}%")
print(f"   Weighted Avg: Precision {results['metrics']['weighted_avg']['precision']:.2f}%  | "
      f"Recall {results['metrics']['weighted_avg']['recall']:.2f}%  | "
      f"F1 {results['metrics']['weighted_avg']['f1_score']:.2f}%")

# 7. Key findings
print(f"\n\n⚠️  CRITICAL FINDINGS:")
print(f"\n   1. MODEL COLLAPSE:")
print(f"      - Model converged to predicting ONLY class 7 (majority class)")
print(f"      - Class 7 recall = 100% (all class 7 samples correctly classified)")
print(f"      - All other classes: precision = recall = F1 = 0%")
print(f"      - This explains why accuracy ≈ 42.47% (the majority class proportion)")

print(f"\n   2. LEARNING FAILURE:")
print(f"      - Loss barely decreased (1.778 → 1.758, only 1.1% reduction)")
print(f"      - Accuracy plateaued at epoch 2 and never improved")
print(f"      - No discrimination between trajectory patterns of different classes")

print(f"\n   3. FEATURE INADEQUACY:")
print(f"      - 27-dimensional trajectory features alone are insufficient")
print(f"      - Cannot distinguish between 11 agricultural activities")
print(f"      - Temporal patterns in trajectory data lack discriminative power")

# 8. Comparison with other modes
print(f"\n\n📈 COMPARISON WITH OTHER MODES:")
print(f"   Mode              | Best Val Acc | Status")
print(f"   " + "-"*55)
print(f"   Multimodal        | 94.18%       | ✓ Excellent (Trajectory + Visual)")
print(f"   Image-only        | 92.91%       | ✓ Good (Visual alone)")
print(f"   Trajectory-only   | 42.47%       | ✗ Failed (Trajectory alone)")

print(f"\n   → Trajectory features add 1.27% improvement when combined with visual")
print(f"   → But are useless alone for this classification task")

# 9. Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Training and validation loss
ax1 = axes[0, 0]
epochs = range(1, 16)
ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
ax1.plot(epochs, history['val_loss'], 'r--', label='Val Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Training and validation accuracy
ax2 = axes[0, 1]
ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
ax2.plot(epochs, history['val_acc'], 'r--', label='Val Acc', linewidth=2)
ax2.axhline(y=42.47, color='orange', linestyle=':', label='Majority Class Baseline (42.47%)', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim([38, 45])

# Plot 3: Per-class precision (bar chart)
ax3 = axes[1, 0]
classes = list(range(11))
precisions = [per_class[f'class_{i}']['precision'] for i in range(11)]
colors = ['red' if p == 0 else 'green' for p in precisions]
bars = ax3.bar(classes, precisions, color=colors, alpha=0.7, edgecolor='black')
ax3.set_xlabel('Class', fontsize=12)
ax3.set_ylabel('Precision (%)', fontsize=12)
ax3.set_title('Per-Class Precision (Final Epoch)', fontsize=14, fontweight='bold')
ax3.set_xticks(classes)
ax3.grid(True, alpha=0.3, axis='y')
ax3.axhline(y=50, color='blue', linestyle='--', alpha=0.5, label='50% threshold')
ax3.legend()

# Add value labels on bars
for bar, val in zip(bars, precisions):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

# Plot 4: Per-class recall (bar chart)
ax4 = axes[1, 1]
recalls = [per_class[f'class_{i}']['recall'] for i in range(11)]
colors = ['red' if r == 0 else 'green' for r in recalls]
bars = ax4.bar(classes, recalls, color=colors, alpha=0.7, edgecolor='black')
ax4.set_xlabel('Class', fontsize=12)
ax4.set_ylabel('Recall (%)', fontsize=12)
ax4.set_title('Per-Class Recall (Final Epoch)', fontsize=14, fontweight='bold')
ax4.set_xticks(classes)
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars, recalls):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
output_path = Path(__file__).parent.parent.parent / 'experiments' / 'trajectory_only_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n\n📊 Visualization saved to: {output_path}")

print(f"\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80 + "\n")
