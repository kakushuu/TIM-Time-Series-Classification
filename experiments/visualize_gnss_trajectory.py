#!/usr/bin/env python3
"""
可视化GNSS轨迹预测
展示真实GNSS轨迹分布 vs 三个模型预测的GNSS轨迹点分布
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
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.insert(0, '/home/research/Agri-MBT/Multimodal-Fusion-with-Attention-Bottlenecks-main/MBT')

from dataloader.av_data import AV_Dataset, TRAJ_COLS
from models.visual_model import AVmodel

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 定义11个类别的颜色
CLASS_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62'
]


def load_or_train_model(mode, checkpoint_path, trainloader, testloader, device, args):
    """加载已有模型或训练新模型"""
    model = AVmodel(
        num_classes=args['num_classes'],
        num_latents=args['num_latent'],
        dim=args['adapter_dim'],
        mode=mode
    )
    model.to(device)

    checkpoint_file = Path(checkpoint_path)

    if checkpoint_file.exists():
        print(f"  ✓ Loading existing checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"  ⚠ No checkpoint found, training new model for {mode}...")
        # 训练模型
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
        criterion = torch.nn.CrossEntropyLoss()

        best_acc = 0
        best_state = None

        for epoch in range(args['num_epochs']):
            # Training
            model.train()
            for traj, imgs, labels in trainloader:
                traj, imgs, labels = traj.to(device), imgs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(traj, imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for traj, imgs, labels in testloader:
                    traj, imgs, labels = traj.to(device), imgs.to(device), labels.to(device)
                    outputs = model(traj, imgs)
                    pred = torch.argmax(outputs, dim=1)
                    correct += (pred == labels).sum().item()
                    total += len(labels)

            acc = 100.0 * correct / total
            if acc > best_acc:
                best_acc = acc
                best_state = model.state_dict().copy()

            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}/{args['num_epochs']}, Val Acc: {acc:.2f}%")

        # Load best model
        model.load_state_dict(best_state)

        # Save checkpoint
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': args['num_epochs'],
            'model_state_dict': best_state,
            'best_val_acc': best_acc,
        }, checkpoint_path)
        print(f"  ✓ Saved checkpoint to {checkpoint_path}")

    model.eval()
    return model


def get_predictions_and_coordinates(models_dict, test_df, test_dataset, device):
    """
    获取模型预测和对应的GNSS坐标

    Returns:
        dict: {
            'ground_truth': {'coords': [...], 'labels': [...]},
            'multimodal': {'coords': [...], 'preds': [...]},
            'image_only': {'coords': [...], 'preds': [...]},
            'trajectory_only': {'coords': [...], 'preds': [...]}
        }
    """
    results = {
        'ground_truth': {'coords': [], 'labels': []},
        'multimodal': {'coords': [], 'preds': []},
        'image_only': {'coords': [], 'preds': []},
        'trajectory_only': {'coords': [], 'preds': []}
    }

    # 提取GNSS坐标
    test_df_subset = test_df.reset_index(drop=True)

    # 创建DataLoader
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for idx, (traj, imgs, labels) in enumerate(test_loader):
            # 获取GNSS坐标
            lon = test_df_subset.loc[idx, '经度']
            lat = test_df_subset.loc[idx, '纬度']

            # 真实标签
            true_label = labels.item()
            results['ground_truth']['coords'].append((lon, lat))
            results['ground_truth']['labels'].append(true_label)

            # 各模型预测
            traj = traj.to(device)
            imgs = imgs.to(device)

            for mode_name, model in models_dict.items():
                outputs = model(traj, imgs)
                pred = torch.argmax(outputs, dim=1).item()
                results[mode_name]['coords'].append((lon, lat))
                results[mode_name]['preds'].append(pred)

    return results


def visualize_gnss_trajectories(results, output_dir):
    """
    可视化GNSS轨迹
    4个子图：真实分布 + 3个模型预测分布
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建4个子图
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))

    titles = [
        'Ground Truth GNSS Trajectory',
        'Multimodal Model Predictions',
        'Image Only Model Predictions',
        'Trajectory Only Model Predictions'
    ]

    data_keys = ['ground_truth', 'multimodal', 'image_only', 'trajectory_only']

    for idx, (ax, title, key) in enumerate(zip(axes.flat, titles, data_keys)):
        data = results[key]

        if key == 'ground_truth':
            labels = data['labels']
        else:
            labels = data['preds']

        coords = np.array(data['coords'])
        lons = coords[:, 0]
        lats = coords[:, 1]
        labels = np.array(labels)

        # 绘制每个类别的点
        for class_id in range(11):
            mask = labels == class_id
            if mask.sum() > 0:
                ax.scatter(lons[mask], lats[mask],
                          c=CLASS_COLORS[class_id],
                          label=f'Class {class_id}',
                          s=15, alpha=0.6, edgecolors='black', linewidth=0.3)

        ax.set_xlabel('Longitude (°)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latitude (°)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

        # 添加图例（只在第一个子图显示）
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9, ncol=2,
                     framealpha=0.9, edgecolor='black')

        # 设置相同的坐标轴范围
        ax.set_xlim([min(results['ground_truth']['coords'])[0] - 0.0001,
                    max(results['ground_truth']['coords'])[0] + 0.0001])
        ax.set_ylim([min(results['ground_truth']['coords'], key=lambda x: x[1])[1] - 0.0001,
                    max(results['ground_truth']['coords'], key=lambda x: x[1])[1] + 0.0001])

    plt.suptitle('GNSS Trajectory Distribution: Ground Truth vs Model Predictions\n(11 Agricultural Activity Classes)',
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()

    save_path = output_dir / 'gnss_trajectory_predictions.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

    # 创建第二张图：按类别统计每个区域的预测准确率
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))

    for idx, (ax, title, key) in enumerate(zip(axes.flat, titles, data_keys)):
        data = results[key]

        if key == 'ground_truth':
            labels = data['labels']
            marker_size = 20
        else:
            labels = data['preds']
            marker_size = 20

        coords = np.array(data['coords'])
        lons = coords[:, 0]
        lats = coords[:, 1]
        labels = np.array(labels)

        # 使用六边形分箱显示密度
        if key != 'ground_truth':
            # 计算预测正确/错误的点
            true_labels = np.array(results['ground_truth']['labels'])
            correct = labels == true_labels

            # 正确的预测
            ax.scatter(lons[correct], lats[correct],
                      c='green', s=15, alpha=0.5, label='Correct', marker='o')

            # 错误的预测
            ax.scatter(lons[~correct], lats[~correct],
                      c='red', s=15, alpha=0.5, label='Wrong', marker='x')

            # 计算准确率
            acc = correct.sum() / len(correct) * 100
            ax.set_title(f'{title}\nAccuracy: {acc:.2f}%', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)
        else:
            # 真实标签按类别着色
            for class_id in range(11):
                mask = labels == class_id
                if mask.sum() > 0:
                    ax.scatter(lons[mask], lats[mask],
                              c=CLASS_COLORS[class_id],
                              label=f'C{class_id}',
                              s=15, alpha=0.6, edgecolors='black', linewidth=0.3)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9, ncol=2)

        ax.set_xlabel('Longitude (°)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latitude (°)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

        # 设置相同的坐标轴范围
        ax.set_xlim([min(results['ground_truth']['coords'])[0] - 0.0001,
                    max(results['ground_truth']['coords'])[0] + 0.0001])
        ax.set_ylim([min(results['ground_truth']['coords'], key=lambda x: x[1])[1] - 0.0001,
                    max(results['ground_truth']['coords'], key=lambda x: x[1])[1] + 0.0001])

    plt.suptitle('GNSS Trajectory Prediction Accuracy Analysis\n(Green=Correct, Red=Wrong)',
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()

    save_path = output_dir / 'gnss_trajectory_accuracy.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

    # 创建第三张图：每个类别的空间分布对比（3个类别一行）
    num_classes_to_show = min(6, 11)  # 只显示前6个类别
    fig, axes = plt.subplots(num_classes_to_show, 4, figsize=(20, 5 * num_classes_to_show))

    for class_id in range(num_classes_to_show):
        for col_idx, (key, title_prefix) in enumerate(zip(data_keys, ['GT', 'Multi', 'Image', 'Traj'])):
            ax = axes[class_id, col_idx]
            data = results[key]

            if key == 'ground_truth':
                labels = np.array(data['labels'])
            else:
                labels = np.array(data['preds'])

            coords = np.array(data['coords'])
            lons = coords[:, 0]
            lats = coords[:, 1]

            # 当前类别的点
            mask = labels == class_id

            # 背景点（灰色）
            ax.scatter(lons[~mask], lats[~mask],
                      c='lightgray', s=5, alpha=0.3, label='Other classes')

            # 当前类别的点
            if mask.sum() > 0:
                ax.scatter(lons[mask], lats[mask],
                          c=CLASS_COLORS[class_id],
                          s=30, alpha=0.8, label=f'Class {class_id}',
                          edgecolors='black', linewidth=0.5)

            ax.set_xlabel('Lon', fontsize=10)
            ax.set_ylabel('Lat', fontsize=10)
            ax.set_title(f'{title_prefix} - Class {class_id} ({mask.sum()} pts)', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

    plt.suptitle('Per-Class GNSS Spatial Distribution Comparison',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    save_path = output_dir / 'gnss_per_class_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    print("\n" + "="*70)
    print("GNSS Trajectory Prediction Visualization")
    print("="*70)

    # 配置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n→ Using device: {device}")

    data_csv = '/home/research/Agri-MBT/data/aligned_output/aligned_data.csv'
    data_dir = '/home/research/Agri-MBT/'
    output_dir = '/home/research/Agri-MBT/experiments/visualizations/gnss_trajectory'

    # 训练参数
    args = {
        'num_classes': 11,
        'num_latent': 4,
        'adapter_dim': 8,
        'lr': 3e-4,
        'num_epochs': 15,
        'batch_size': 8
    }

    # 加载数据
    print("\n→ Loading dataset...")
    full_df = pd.read_csv(data_csv)
    print(f"  Total samples: {len(full_df)}")

    # 划分训练集和测试集
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
    split = int(len(full_df) * 0.8)
    train_df = full_df.iloc[:split]
    test_df = full_df.iloc[split:]

    # 计算归一化统计量
    traj_vals = train_df[TRAJ_COLS].values.astype('float32')
    traj_mean = traj_vals.mean(axis=0)
    traj_std = traj_vals.std(axis=0) + 1e-6

    # 创建数据集
    train_dataset = AV_Dataset(train_df, data_dir=data_dir,
                               traj_mean=traj_mean, traj_std=traj_std)
    test_dataset = AV_Dataset(test_df, data_dir=data_dir,
                              traj_mean=traj_mean, traj_std=traj_std)

    trainloader = DataLoader(train_dataset, batch_size=args['batch_size'],
                            shuffle=True, num_workers=4)
    testloader = DataLoader(test_dataset, batch_size=args['batch_size'],
                           shuffle=False, num_workers=4)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    # 加载/训练模型
    print("\n→ Loading/Training models...")
    checkpoint_dir = Path('/home/research/Agri-MBT/experiments/checkpoints')

    models_dict = {}
    for mode in ['multimodal', 'image_only', 'trajectory_only']:
        checkpoint_path = checkpoint_dir / f'{mode}_best.pth'
        print(f"\n  Processing {mode}...")
        models_dict[mode] = load_or_train_model(mode, checkpoint_path,
                                                trainloader, testloader,
                                                device, args)

    # 获取预测和坐标
    print("\n→ Getting predictions and coordinates...")
    results = get_predictions_and_coordinates(models_dict, test_df,
                                              test_dataset, device)

    # 统计
    print("\n→ Prediction Statistics:")
    for mode in ['multimodal', 'image_only', 'trajectory_only']:
        true_labels = np.array(results['ground_truth']['labels'])
        pred_labels = np.array(results[mode]['preds'])
        acc = (true_labels == pred_labels).sum() / len(true_labels) * 100
        print(f"  {mode:20s}: {acc:.2f}% accuracy")

    # 可视化
    print("\n→ Creating visualizations...")
    visualize_gnss_trajectories(results, output_dir)

    print("\n" + "="*70)
    print("✓ Visualization complete!")
    print(f"  Output directory: {output_dir}")
    print("\n📊 Generated visualizations:")
    print("  1. gnss_trajectory_predictions.png - Overall distribution comparison")
    print("  2. gnss_trajectory_accuracy.png - Accuracy analysis (correct/wrong)")
    print("  3. gnss_per_class_distribution.png - Per-class spatial distribution")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
