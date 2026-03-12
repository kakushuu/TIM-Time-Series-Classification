#!/usr/bin/env python3
"""
可视化三个模型对轨迹的预测结果
显示每个模型对不同样本的预测概率分布，以及真实标签
"""
import matplotlib
matplotlib.use('Agg')

import sys
import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.insert(0, '/home/research/Agri-MBT/Multimodal-Fusion-with-Attention-Bottlenecks-main/MBT')

from dataloader.av_data import AV_Dataset, TRAJ_COLS
from models.visual_model import AVmodel

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 类别名称（根据实际数据定义）
CLASS_NAMES = [
    'Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4',
    'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9', 'Class 10'
]

def load_model(mode, checkpoint_path, device='cuda:0'):
    """加载训练好的模型"""
    model = AVmodel(
        num_classes=11,
        num_latents=4,
        dim=8,
        mode=mode
    )

    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"⚠ Checkpoint not found at {checkpoint_path}, using untrained model")

    model.to(device)
    model.eval()
    return model


def get_sample_predictions(models_dict, test_loader, device, num_samples=3):
    """
    获取多个模型对样本的预测

    Returns:
        dict: {
            'ground_truth': [label1, label2, ...],
            'multimodal': {'probs': [...], 'preds': [...]},
            'image_only': {'probs': [...], 'preds': [...]},
            'trajectory_only': {'probs': [...], 'preds': [...]}
        }
    """
    results = {
        'ground_truth': [],
        'multimodal': {'probs': [], 'preds': []},
        'image_only': {'probs': [], 'preds': []},
        'trajectory_only': {'probs': [], 'preds': []}
    }

    sample_count = 0
    with torch.no_grad():
        for traj, imgs, labels in test_loader:
            if sample_count >= num_samples:
                break

            traj = traj.to(device)
            imgs = imgs.to(device)

            # 获取每个模型的预测
            for mode_name, model in models_dict.items():
                outputs = model(traj, imgs)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                results[mode_name]['probs'].extend(probs.cpu().numpy())
                results[mode_name]['preds'].extend(preds.cpu().numpy())

            results['ground_truth'].extend(labels.numpy())
            sample_count += len(labels)

    return results


def visualize_predictions(results, output_dir):
    """
    可视化预测结果
    绘制4个子图：
    1-3: 三个模型的预测概率分布（热力图）
    4: 真实标签
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_samples = len(results['ground_truth'])
    num_classes = 11

    # 为每个样本创建一个独立的图
    for sample_idx in range(num_samples):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Sample {sample_idx + 1} - Model Predictions vs Ground Truth',
                     fontsize=16, fontweight='bold', y=0.995)

        models = ['multimodal', 'image_only', 'trajectory_only']
        titles = ['Multimodal Model', 'Image Only Model', 'Trajectory Only Model']

        # 绘制三个模型的预测概率分布
        for idx, (model_name, title) in enumerate(zip(models, titles)):
            ax = axes[idx // 2, idx % 2]

            probs = results[model_name]['probs'][sample_idx]
            pred_label = results[model_name]['preds'][sample_idx]
            true_label = results['ground_truth'][sample_idx]

            # 创建概率条形图
            colors = ['#3498db'] * num_classes
            colors[pred_label] = '#e74c3c'  # 预测类别用红色
            colors[true_label] = '#2ecc71' if pred_label == true_label else '#f39c12'  # 真实标签

            bars = ax.bar(range(num_classes), probs, color=colors, alpha=0.8, edgecolor='black')

            # 添加数值标签
            for i, (bar, prob) in enumerate(zip(bars, probs)):
                if prob > 0.05:  # 只显示概率 > 5% 的标签
                    ax.text(i, prob + 0.02, f'{prob:.2f}', ha='center', va='bottom',
                           fontsize=8, fontweight='bold')

            ax.set_xlabel('Class Label', fontsize=11, fontweight='bold')
            ax.set_ylabel('Prediction Probability', fontsize=11, fontweight='bold')
            ax.set_title(f'{title}\nPredicted: {pred_label}, True: {true_label}',
                        fontsize=12, fontweight='bold')
            ax.set_xticks(range(num_classes))
            ax.set_xticklabels([f'C{i}' for i in range(num_classes)], fontsize=9)
            ax.set_ylim([0, 1.1])
            ax.grid(axis='y', alpha=0.3)

            # 添加图例
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#e74c3c', edgecolor='black', label='Predicted'),
                Patch(facecolor='#2ecc71', edgecolor='black', label='Correct'),
                Patch(facecolor='#f39c12', edgecolor='black', label='True (wrong pred)'),
                Patch(facecolor='#3498db', edgecolor='black', label='Other classes')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        # 第4个子图：真实标签的one-hot编码
        ax = axes[1, 1]
        true_label = results['ground_truth'][sample_idx]
        one_hot = np.zeros(num_classes)
        one_hot[true_label] = 1.0

        ax.bar(range(num_classes), one_hot, color='#9b59b6', alpha=0.8, edgecolor='black')
        ax.set_xlabel('Class Label', fontsize=11, fontweight='bold')
        ax.set_ylabel('Ground Truth', fontsize=11, fontweight='bold')
        ax.set_title(f'Ground Truth Label: {true_label}', fontsize=12, fontweight='bold')
        ax.set_xticks(range(num_classes))
        ax.set_xticklabels([f'C{i}' for i in range(num_classes)], fontsize=9)
        ax.set_ylim([0, 1.2])
        ax.grid(axis='y', alpha=0.3)

        # 标注真实类别
        ax.text(true_label, 1.05, f'True\nClass {true_label}', ha='center', va='bottom',
               fontsize=10, fontweight='bold', color='#9b59b6')

        plt.tight_layout()
        save_path = output_dir / f'sample_{sample_idx + 1}_predictions.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()

    # 创建综合对比图（所有样本在一张图中）
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for sample_idx in range(num_samples):
        models = ['multimodal', 'image_only', 'trajectory_only']
        titles = ['Multimodal', 'Image Only', 'Trajectory Only', 'Ground Truth']

        # 绘制三个模型的预测
        for col_idx in range(3):
            ax = axes[sample_idx, col_idx]
            model_name = models[col_idx]

            probs = results[model_name]['probs'][sample_idx]
            pred_label = results[model_name]['preds'][sample_idx]
            true_label = results['ground_truth'][sample_idx]

            # 热力图形式
            probs_2d = probs.reshape(1, -1)
            sns.heatmap(probs_2d, annot=True, fmt='.2f', cmap='YlOrRd',
                       xticklabels=[f'C{i}' for i in range(num_classes)],
                       yticklabels=[''], cbar=True, ax=ax,
                       vmin=0, vmax=1, annot_kws={'size': 9, 'weight': 'bold'})

            correct = '✓' if pred_label == true_label else '✗'
            ax.set_title(f'{titles[col_idx]}\nPred: {pred_label}, True: {true_label} {correct}',
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Class', fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f'Sample {sample_idx + 1}', fontsize=11, fontweight='bold')

        # 第4列：真实标签
        ax = axes[sample_idx, 3]
        true_label = results['ground_truth'][sample_idx]
        one_hot = np.zeros(num_classes)
        one_hot[true_label] = 1.0
        one_hot_2d = one_hot.reshape(1, -1)

        sns.heatmap(one_hot_2d, annot=True, fmt='.0f', cmap='Purples',
                   xticklabels=[f'C{i}' for i in range(num_classes)],
                   yticklabels=[''], cbar=False, ax=ax,
                   vmin=0, vmax=1, annot_kws={'size': 9, 'weight': 'bold'})

        ax.set_title(f'Ground Truth: {true_label}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Class', fontsize=10)

    plt.suptitle('Model Predictions Comparison Across Samples',
                 fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    save_path = output_dir / 'all_samples_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    print("\n" + "="*70)
    print("Trajectory Prediction Visualization")
    print("="*70)

    # 配置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_csv = '/home/research/Agri-MBT/data/aligned_output/aligned_data.csv'
    data_dir = '/home/research/Agri-MBT/'
    output_dir = '/home/research/Agri-MBT/experiments/visualizations/trajectory_predictions'

    # 加载数据
    print("\n→ Loading dataset...")
    full_df = pd.read_csv(data_csv)
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 使用20%作为测试集
    split = int(len(full_df) * 0.8)
    test_df = full_df.iloc[split:]

    # 计算归一化统计量
    traj_vals = full_df[TRAJ_COLS].values.astype('float32')
    traj_mean = traj_vals.mean(axis=0)
    traj_std = traj_vals.std(axis=0) + 1e-6

    test_dataset = AV_Dataset(test_df, data_dir=data_dir,
                              traj_mean=traj_mean, traj_std=traj_std)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"  Test samples: {len(test_dataset)}")

    # 加载模型
    print("\n→ Loading models...")
    checkpoint_dir = Path('/home/research/Agri-MBT/experiments/checkpoints')

    models_dict = {}
    for mode in ['multimodal', 'image_only', 'trajectory_only']:
        checkpoint_path = checkpoint_dir / f'{mode}_best.pth'
        try:
            models_dict[mode] = load_model(mode, checkpoint_path, device)
        except Exception as e:
            print(f"⚠ Failed to load {mode} model: {e}")
            # 使用未训练的模型作为fallback
            models_dict[mode] = load_model(mode, checkpoint_path, device)

    # 获取预测
    print("\n→ Getting predictions for 3 samples...")
    results = get_sample_predictions(models_dict, test_loader, device, num_samples=3)

    # 可视化
    print("\n→ Creating visualizations...")
    visualize_predictions(results, output_dir)

    print("\n" + "="*70)
    print("✓ Visualization complete!")
    print(f"  Output directory: {output_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
