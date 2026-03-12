#!/usr/bin/env python3
"""
使用seaborn绘制三个模型的预测性能对比图
展示三个轨迹（multimodal, image_only, trajectory_only）的预测结果以及真实标签分布
"""
import matplotlib
matplotlib.use('Agg')

import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 11个类别的名称
CLASS_NAMES = [
    'Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4',
    'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9', 'Class 10'
]

def load_results():
    """加载三个模型的实验结果"""
    results_dir = Path('/home/research/Agri-MBT/experiments')

    with open(results_dir / 'results_multimodal.json', 'r') as f:
        multimodal = json.load(f)

    with open(results_dir / 'results_trajectory_only.json', 'r') as f:
        trajectory_only = json.load(f)

    with open(results_dir / 'results_image_only.json', 'r') as f:
        image_only = json.load(f)

    return multimodal, image_only, trajectory_only


def create_prediction_comparison_plot(multimodal, image_only, trajectory_only, output_dir):
    """
    创建三个模型的预测性能对比图
    包括：
    1. 三个模型对每个类别的F1-Score预测（热力图）
    2. 真实标签的分布（柱状图）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_classes = 11

    # 准备数据
    multimodal_f1 = [multimodal['metrics']['per_class'][f'class_{i}']['f1_score'] for i in range(num_classes)]
    image_f1 = [image_only['metrics']['per_class'][f'class_{i}']['f1_score'] for i in range(num_classes)]
    trajectory_f1 = [trajectory_only['metrics']['per_class'][f'class_{i}']['f1_score'] for i in range(num_classes)]

    # 样本数分布（从已有数据中获取）
    sample_counts = [1152, 295, 682, 2540, 489, 759, 350, 11069, 2053, 799, 5772]

    # ============ 图1：三个模型的F1-Score热力图对比 ============
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Multimodal Model
    ax = axes[0, 0]
    data_2d = np.array(multimodal_f1).reshape(1, -1)
    sns.heatmap(data_2d, annot=True, fmt='.1f', cmap='YlGnBu',
               xticklabels=[f'C{i}' for i in range(num_classes)],
               yticklabels=['F1-Score'], cbar_kws={'label': 'F1-Score (%)'},
               ax=ax, vmin=0, vmax=100, annot_kws={'size': 11, 'weight': 'bold'})
    ax.set_title(f'Multimodal Model (Trajectory + Image)\nMacro F1: {multimodal["metrics"]["macro_avg"]["f1_score"]:.2f}%',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')

    # Image Only Model
    ax = axes[0, 1]
    data_2d = np.array(image_f1).reshape(1, -1)
    sns.heatmap(data_2d, annot=True, fmt='.1f', cmap='YlGnBu',
               xticklabels=[f'C{i}' for i in range(num_classes)],
               yticklabels=['F1-Score'], cbar_kws={'label': 'F1-Score (%)'},
               ax=ax, vmin=0, vmax=100, annot_kws={'size': 11, 'weight': 'bold'})
    ax.set_title(f'Image Only Model\nMacro F1: {image_only["metrics"]["macro_avg"]["f1_score"]:.2f}%',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')

    # Trajectory Only Model
    ax = axes[1, 0]
    data_2d = np.array(trajectory_f1).reshape(1, -1)
    sns.heatmap(data_2d, annot=True, fmt='.1f', cmap='YlGnBu',
               xticklabels=[f'C{i}' for i in range(num_classes)],
               yticklabels=['F1-Score'], cbar_kws={'label': 'F1-Score (%)'},
               ax=ax, vmin=0, vmax=100, annot_kws={'size': 11, 'weight': 'bold'})
    ax.set_title(f'Trajectory Only Model\nMacro F1: {trajectory_only["metrics"]["macro_avg"]["f1_score"]:.2f}%',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')

    # 真实标签分布
    ax = axes[1, 1]
    bars = ax.bar(range(num_classes), sample_counts, color='#9b59b6', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Class Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('Ground Truth Label Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels([f'C{i}' for i in range(num_classes)], fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for i, (bar, count) in enumerate(zip(bars, sample_counts)):
        if count > 1000:
            ax.text(i, count + 200, f'{count}', ha='center', va='bottom',
                   fontsize=8, fontweight='bold', rotation=45)
        else:
            ax.text(i, count + 50, f'{count}', ha='center', va='bottom',
                   fontsize=8, fontweight='bold')

    plt.suptitle('Model Prediction Performance Comparison\n(11 Agricultural Activity Classes)',
                 fontsize=18, fontweight='bold', y=1.0)
    plt.tight_layout()
    save_path = output_dir / 'model_predictions_heatmap.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

    # ============ 图2：三个模型的F1-Score对比柱状图 ============
    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(num_classes)
    width = 0.25

    bars1 = ax.bar(x - width, multimodal_f1, width, label='Multimodal (Trajectory+Image)',
                   color='#2ecc71', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, image_f1, width, label='Image Only',
                   color='#3498db', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, trajectory_f1, width, label='Trajectory Only',
                   color='#e74c3c', alpha=0.8, edgecolor='black')

    ax.set_xlabel('Class Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('F1-Score (%)', fontsize=13, fontweight='bold')
    ax.set_title('Per-Class F1-Score Prediction Performance\nThree Models Comparison',
                fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Class {i}' for i in range(num_classes)], fontsize=11)
    ax.legend(fontsize=12, loc='upper right')
    ax.set_ylim([0, 110])
    ax.grid(axis='y', alpha=0.3)

    # 添加数值标签（仅显示多模态和图像单模态）
    for i, (v1, v2) in enumerate(zip(multimodal_f1, image_f1)):
        if v1 > 5:
            ax.text(i - width, v1 + 2, f'{v1:.1f}', ha='center', va='bottom',
                   fontsize=9, fontweight='bold', rotation=45)
        if v2 > 5:
            ax.text(i, v2 + 2, f'{v2:.1f}', ha='center', va='bottom',
                   fontsize=9, fontweight='bold', rotation=45)

    plt.tight_layout()
    save_path = output_dir / 'model_predictions_barplot.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

    # ============ 图3：综合对比（包含真实分布和预测性能）============
    fig = plt.figure(figsize=(20, 10))

    # 创建网格布局
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 左上：真实标签分布
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(range(num_classes), sample_counts, color='#9b59b6', alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Class', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Sample Count', fontsize=11, fontweight='bold')
    ax1.set_title('Ground Truth Distribution', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(num_classes))
    ax1.set_xticklabels([f'C{i}' for i in range(num_classes)], fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # 右上：三个模型的平均性能对比
    ax2 = fig.add_subplot(gs[0, 1])
    models = ['Multimodal', 'Image Only', 'Trajectory Only']
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    avg_f1 = [
        multimodal['metrics']['macro_avg']['f1_score'],
        image_only['metrics']['macro_avg']['f1_score'],
        trajectory_only['metrics']['macro_avg']['f1_score']
    ]

    bars = ax2.bar(models, avg_f1, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Macro F1-Score (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Average Prediction Performance', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 100])
    ax2.grid(axis='y', alpha=0.3)

    for i, (bar, v) in enumerate(zip(bars, avg_f1)):
        ax2.text(i, v + 2, f'{v:.2f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    # 右上角：准确率对比
    ax3 = fig.add_subplot(gs[0, 2])
    accuracies = [
        multimodal['best_val_acc'],
        image_only['best_val_acc'],
        trajectory_only['best_val_acc']
    ]

    bars = ax3.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Validation Accuracy', fontsize=13, fontweight='bold')
    ax3.set_ylim([0, 100])
    ax3.grid(axis='y', alpha=0.3)

    for i, (bar, v) in enumerate(zip(bars, accuracies)):
        ax3.text(i, v + 2, f'{v:.2f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    # 底部：详细的热力图对比
    # Multimodal
    ax4 = fig.add_subplot(gs[1, 0])
    data_2d = np.array([multimodal_f1])
    sns.heatmap(data_2d, annot=True, fmt='.1f', cmap='YlGnBu',
               xticklabels=[f'C{i}' for i in range(num_classes)],
               yticklabels=['F1'], cbar=True, ax=ax4,
               vmin=0, vmax=100, annot_kws={'size': 9, 'weight': 'bold'})
    ax4.set_title('Multimodal Predictions', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Class', fontsize=10)

    # Image Only
    ax5 = fig.add_subplot(gs[1, 1])
    data_2d = np.array([image_f1])
    sns.heatmap(data_2d, annot=True, fmt='.1f', cmap='YlGnBu',
               xticklabels=[f'C{i}' for i in range(num_classes)],
               yticklabels=['F1'], cbar=True, ax=ax5,
               vmin=0, vmax=100, annot_kws={'size': 9, 'weight': 'bold'})
    ax5.set_title('Image Only Predictions', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Class', fontsize=10)

    # Trajectory Only
    ax6 = fig.add_subplot(gs[1, 2])
    data_2d = np.array([trajectory_f1])
    sns.heatmap(data_2d, annot=True, fmt='.1f', cmap='YlGnBu',
               xticklabels=[f'C{i}' for i in range(num_classes)],
               yticklabels=['F1'], cbar=True, ax=ax6,
               vmin=0, vmax=100, annot_kws={'size': 9, 'weight': 'bold'})
    ax6.set_title('Trajectory Only Predictions', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Class', fontsize=10)

    plt.suptitle('Comprehensive Model Prediction Analysis\n3 Models × 11 Classes',
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_path = output_dir / 'model_predictions_comprehensive.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

    # ============ 图4：每个类别的详细预测分析 ============
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Per-Class Prediction Analysis: Precision, Recall, and F1-Score',
                 fontsize=18, fontweight='bold', y=0.995)

    metrics = ['precision', 'recall', 'f1_score']
    metric_names = ['Precision (%)', 'Recall (%)', 'F1-Score (%)']

    for row_idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        # Multimodal
        ax = axes[row_idx, 0]
        values = [multimodal['metrics']['per_class'][f'class_{i}'][metric] for i in range(num_classes)]
        sns.barplot(x=list(range(num_classes)), y=values, ax=ax, color='#2ecc71', alpha=0.8)
        ax.set_title(f'Multimodal\n{metric_name}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Class', fontsize=9)
        ax.set_ylabel(metric_name, fontsize=9)
        ax.set_ylim([0, 110])
        ax.grid(axis='y', alpha=0.3)

        # Image Only
        ax = axes[row_idx, 1]
        values = [image_only['metrics']['per_class'][f'class_{i}'][metric] for i in range(num_classes)]
        sns.barplot(x=list(range(num_classes)), y=values, ax=ax, color='#3498db', alpha=0.8)
        ax.set_title(f'Image Only\n{metric_name}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Class', fontsize=9)
        ax.set_ylabel(metric_name, fontsize=9)
        ax.set_ylim([0, 110])
        ax.grid(axis='y', alpha=0.3)

        # Trajectory Only
        ax = axes[row_idx, 2]
        values = [trajectory_only['metrics']['per_class'][f'class_{i}'][metric] for i in range(num_classes)]
        sns.barplot(x=list(range(num_classes)), y=values, ax=ax, color='#e74c3c', alpha=0.8)
        ax.set_title(f'Trajectory Only\n{metric_name}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Class', fontsize=9)
        ax.set_ylabel(metric_name, fontsize=9)
        ax.set_ylim([0, 110])
        ax.grid(axis='y', alpha=0.3)

        # Ground Truth (Sample Count) - only in first row
        if row_idx == 0:
            ax = axes[row_idx, 3]
            ax.bar(range(num_classes), sample_counts, color='#9b59b6', alpha=0.8, edgecolor='black')
            ax.set_xlabel('Class', fontsize=9)
            ax.set_ylabel('Sample Count', fontsize=9)
            ax.set_title('Ground Truth\nDistribution', fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        else:
            axes[row_idx, 3].axis('off')

    plt.tight_layout()
    save_path = output_dir / 'model_predictions_per_class_detailed.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    print("\n" + "="*70)
    print("Trajectory Prediction Visualization (Using Experiment Results)")
    print("="*70)

    # 加载结果
    print("\n→ Loading experiment results...")
    multimodal, image_only, trajectory_only = load_results()

    print(f"  Multimodal:     {multimodal['best_val_acc']:.2f}% accuracy")
    print(f"  Image Only:     {image_only['best_val_acc']:.2f}% accuracy")
    print(f"  Trajectory Only: {trajectory_only['best_val_acc']:.2f}% accuracy")

    # 创建可视化
    output_dir = '/home/research/Agri-MBT/experiments/visualizations/trajectory_predictions'
    print(f"\n→ Creating visualizations in {output_dir}...")
    create_prediction_comparison_plot(multimodal, image_only, trajectory_only, output_dir)

    print("\n" + "="*70)
    print("✓ Visualization complete!")
    print(f"  Output directory: {output_dir}")
    print("\n📊 Generated visualizations:")
    print("  1. model_predictions_heatmap.png - Heatmap comparison of F1-scores")
    print("  2. model_predictions_barplot.png - Bar plot comparison")
    print("  3. model_predictions_comprehensive.png - Comprehensive analysis")
    print("  4. model_predictions_per_class_detailed.png - Detailed per-class metrics")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
