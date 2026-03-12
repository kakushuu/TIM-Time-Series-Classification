#!/usr/bin/env python3
"""
可视化三个实验的对比结果
"""
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示服务器问题

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 读取结果文件
results_dir = Path('/home/research/Agri-MBT/experiments')

with open(results_dir / 'results_multimodal.json', 'r') as f:
    multimodal = json.load(f)

with open(results_dir / 'results_trajectory_only.json', 'r') as f:
    trajectory_only = json.load(f)

with open(results_dir / 'results_image_only.json', 'r') as f:
    image_only = json.load(f)

# 创建输出目录
output_dir = Path('/home/research/Agri-MBT/experiments/visualizations')
output_dir.mkdir(exist_ok=True)

# 1. 整体性能对比柱状图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

models = ['Multimodal\n(Trajectory+Image)', 'Image Only', 'Trajectory Only']
colors = ['#2ecc71', '#3498db', '#e74c3c']

# 准确率
accuracies = [
    multimodal['best_val_acc'],
    image_only['best_val_acc'],
    trajectory_only['best_val_acc']
]
axes[0].bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
axes[0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
axes[0].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylim([0, 100])
for i, v in enumerate(accuracies):
    axes[0].text(i, v + 1, f'{v:.2f}%', ha='center', fontweight='bold', fontsize=11)

# Macro F1
macro_f1 = [
    multimodal['metrics']['macro_avg']['f1_score'],
    image_only['metrics']['macro_avg']['f1_score'],
    trajectory_only['metrics']['macro_avg']['f1_score']
]
axes[1].bar(models, macro_f1, color=colors, alpha=0.8, edgecolor='black')
axes[1].set_ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
axes[1].set_title('Macro F1-Score Comparison', fontsize=14, fontweight='bold')
axes[1].set_ylim([0, 100])
for i, v in enumerate(macro_f1):
    axes[1].text(i, v + 1, f'{v:.2f}%', ha='center', fontweight='bold', fontsize=11)

# Weighted F1
weighted_f1 = [
    multimodal['metrics']['weighted_avg']['f1_score'],
    image_only['metrics']['weighted_avg']['f1_score'],
    trajectory_only['metrics']['weighted_avg']['f1_score']
]
axes[2].bar(models, weighted_f1, color=colors, alpha=0.8, edgecolor='black')
axes[2].set_ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
axes[2].set_title('Weighted F1-Score Comparison', fontsize=14, fontweight='bold')
axes[2].set_ylim([0, 100])
for i, v in enumerate(weighted_f1):
    axes[2].text(i, v + 1, f'{v:.2f}%', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / 'overall_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'overall_comparison.png'}")

# 2. 每类F1-Score对比
fig, ax = plt.subplots(figsize=(14, 7))

classes = [f'Class {i}' for i in range(11)]
x = np.arange(len(classes))
width = 0.25

multimodal_f1 = [multimodal['metrics']['per_class'][f'class_{i}']['f1_score'] for i in range(11)]
image_f1 = [image_only['metrics']['per_class'][f'class_{i}']['f1_score'] for i in range(11)]
trajectory_f1 = [trajectory_only['metrics']['per_class'][f'class_{i}']['f1_score'] for i in range(11)]

bars1 = ax.bar(x - width, multimodal_f1, width, label='Multimodal', color='#2ecc71', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x, image_f1, width, label='Image Only', color='#3498db', alpha=0.8, edgecolor='black')
bars3 = ax.bar(x + width, trajectory_f1, width, label='Trajectory Only', color='#e74c3c', alpha=0.8, edgecolor='black')

ax.set_xlabel('Class', fontsize=13, fontweight='bold')
ax.set_ylabel('F1-Score (%)', fontsize=13, fontweight='bold')
ax.set_title('Per-Class F1-Score Comparison Across Models', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=11)
ax.legend(fontsize=11, loc='upper right')
ax.set_ylim([0, 110])
ax.grid(axis='y', alpha=0.3)

# 添加数值标签（仅显示多模态和图像单模态，轨迹单模态大部分为0）
for i, (v1, v2) in enumerate(zip(multimodal_f1, image_f1)):
    if v1 > 5:
        ax.text(i - width, v1 + 1, f'{v1:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    if v2 > 5:
        ax.text(i, v2 + 1, f'{v2:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'per_class_f1_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'per_class_f1_comparison.png'}")

# 3. 每类Precision和Recall对比（热力图）
fig, axes = plt.subplots(3, 3, figsize=(18, 14))

# 准备数据
models_data = {
    'Multimodal': multimodal,
    'Image Only': image_only,
    'Trajectory Only': trajectory_only
}

metrics = ['precision', 'recall', 'f1_score']
metric_names = ['Precision (%)', 'Recall (%)', 'F1-Score (%)']

for row_idx, metric in enumerate(metrics):
    for col_idx, (model_name, model_data) in enumerate(models_data.items()):
        values = [model_data['metrics']['per_class'][f'class_{i}'][metric] for i in range(11)]
        values_array = np.array(values).reshape(1, -1)

        sns.heatmap(values_array, annot=True, fmt='.1f', cmap='YlGnBu',
                   xticklabels=[f'C{i}' for i in range(11)],
                   yticklabels=[''],
                   cbar_kws={'label': metric_names[row_idx]},
                   ax=axes[row_idx, col_idx],
                   vmin=0, vmax=100,
                   annot_kws={'size': 10, 'weight': 'bold'})

        if row_idx == 0:
            axes[row_idx, col_idx].set_title(f'{model_name}', fontsize=14, fontweight='bold')
        if col_idx == 0:
            axes[row_idx, col_idx].set_ylabel(metric_names[row_idx], fontsize=12, fontweight='bold')

plt.suptitle('Per-Class Metrics Heatmap Comparison', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'metrics_heatmap.png'}")

# 4. 训练曲线对比（仅多模态和图像单模态，轨迹单模态无意义）
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 多模态训练曲线
epochs = range(1, len(multimodal['history']['train_loss']) + 1)
axes[0].plot(epochs, multimodal['history']['train_acc'], 'o-', label='Train Acc',
             color='#2ecc71', linewidth=2, markersize=6)
axes[0].plot(epochs, multimodal['history']['val_acc'], 's-', label='Val Acc',
             color='#e74c3c', linewidth=2, markersize=6)
axes[0].axhline(y=multimodal['best_val_acc'], color='red', linestyle='--',
                label=f"Best Val: {multimodal['best_val_acc']:.2f}%", alpha=0.7)
axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
axes[0].set_title('Multimodal Training Progress', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# 图像单模态训练曲线
epochs = range(1, len(image_only['history']['train_loss']) + 1)
axes[1].plot(epochs, image_only['history']['train_acc'], 'o-', label='Train Acc',
             color='#3498db', linewidth=2, markersize=6)
axes[1].plot(epochs, image_only['history']['val_acc'], 's-', label='Val Acc',
             color='#e74c3c', linewidth=2, markersize=6)
axes[1].axhline(y=image_only['best_val_acc'], color='red', linestyle='--',
                label=f"Best Val: {image_only['best_val_acc']:.2f}%", alpha=0.7)
axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
axes[1].set_title('Image Only Training Progress', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'training_curves.png'}")

# 5. 类别样本数与F1-Score关系图
fig, ax = plt.subplots(figsize=(12, 7))

# 样本数（从结果中获取）
sample_counts = [1152, 295, 682, 2540, 489, 759, 350, 11069, 2053, 799, 5772]

ax.scatter(sample_counts, multimodal_f1, s=150, alpha=0.7, color='#2ecc71',
           label='Multimodal', edgecolors='black', linewidth=2, marker='o')
ax.scatter(sample_counts, image_f1, s=150, alpha=0.7, color='#3498db',
           label='Image Only', edgecolors='black', linewidth=2, marker='s')
ax.scatter(sample_counts, trajectory_f1, s=150, alpha=0.7, color='#e74c3c',
           label='Trajectory Only', edgecolors='black', linewidth=2, marker='^')

# 添加类别标签
for i, (x, y1, y2) in enumerate(zip(sample_counts, multimodal_f1, image_f1)):
    ax.annotate(f'C{i}', (x, max(y1, y2) + 3), ha='center', fontsize=9, fontweight='bold')

ax.set_xlabel('Number of Training Samples', fontsize=13, fontweight='bold')
ax.set_ylabel('F1-Score (%)', fontsize=13, fontweight='bold')
ax.set_title('Class Sample Count vs F1-Score', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xscale('log')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'sample_count_vs_f1.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'sample_count_vs_f1.png'}")

# 6. 综合对比雷达图
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

categories = ['Accuracy', 'Macro F1', 'Weighted F1',
              'Avg Precision\n(Macro)', 'Avg Recall\n(Macro)', 'Best Class F1']
N = len(categories)

# 准备数据
multimodal_values = [
    multimodal['best_val_acc'],
    multimodal['metrics']['macro_avg']['f1_score'],
    multimodal['metrics']['weighted_avg']['f1_score'],
    multimodal['metrics']['macro_avg']['precision'],
    multimodal['metrics']['macro_avg']['recall'],
    max([multimodal['metrics']['per_class'][f'class_{i}']['f1_score'] for i in range(11)])
]

image_values = [
    image_only['best_val_acc'],
    image_only['metrics']['macro_avg']['f1_score'],
    image_only['metrics']['weighted_avg']['f1_score'],
    image_only['metrics']['macro_avg']['precision'],
    image_only['metrics']['macro_avg']['recall'],
    max([image_only['metrics']['per_class'][f'class_{i}']['f1_score'] for i in range(11)])
]

trajectory_values = [
    trajectory_only['best_val_acc'],
    trajectory_only['metrics']['macro_avg']['f1_score'],
    trajectory_only['metrics']['weighted_avg']['f1_score'],
    trajectory_only['metrics']['macro_avg']['precision'],
    trajectory_only['metrics']['macro_avg']['recall'],
    max([trajectory_only['metrics']['per_class'][f'class_{i}']['f1_score'] for i in range(11)])
]

# 闭合雷达图
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
multimodal_values += multimodal_values[:1]
image_values += image_values[:1]
trajectory_values += trajectory_values[:1]
angles += angles[:1]

ax.plot(angles, multimodal_values, 'o-', linewidth=2, label='Multimodal', color='#2ecc71')
ax.fill(angles, multimodal_values, alpha=0.25, color='#2ecc71')
ax.plot(angles, image_values, 's-', linewidth=2, label='Image Only', color='#3498db')
ax.fill(angles, image_values, alpha=0.25, color='#3498db')
ax.plot(angles, trajectory_values, '^-', linewidth=2, label='Trajectory Only', color='#e74c3c')
ax.fill(angles, trajectory_values, alpha=0.25, color='#e74c3c')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax.set_ylim(0, 100)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax.set_title('Model Performance Radar Chart', fontsize=15, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(output_dir / 'radar_chart.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'radar_chart.png'}")

print("\n" + "="*60)
print("✓ All visualizations saved to:")
print(f"  {output_dir}")
print("="*60)

# 打印总结
print("\n📊 Visualization Summary:")
print("  1. overall_comparison.png - Overall performance comparison")
print("  2. per_class_f1_comparison.png - Per-class F1-score comparison")
print("  3. metrics_heatmap.png - Precision/Recall/F1 heatmap")
print("  4. training_curves.png - Training progress curves")
print("  5. sample_count_vs_f1.png - Sample count vs F1-score")
print("  6. radar_chart.png - Comprehensive radar chart")
