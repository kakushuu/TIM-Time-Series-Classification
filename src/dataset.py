#!/usr/bin/env python3
"""
农机多模态数据集加载器
AgriMultimodalDataset: 从 aligned_data.csv 加载 T=5 滑动窗口数据

输出:
  video_tensor: (T, 3, 224, 224)  — T帧 RGB 图像
  gnss_tensor:  (7,)              — 窗口中点帧的7维GNSS特征
  label:        int               — 0~10 农业活动类别
"""

import json
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

# 项目根目录（相对于本文件的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# GNSS 特征列名
GNSS_COLS = ['经度', '纬度', '速度', '深度', '方向角', '间距(米)', '类型']

# ImageNet 归一化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class AgriMultimodalDataset(Dataset):
    """
    农机多模态数据集

    从 aligned_data.csv 提取 T=5 的连续帧窗口，每条样本包含：
    - T帧 RGB 视频帧（已对齐的 jpg 图片）
    - 窗口中点帧的 7维 GNSS 特征（经过 z-score 归一化）
    - 农业活动类别标签 (0~10)

    滑动窗口逻辑：
      对同一 video_file 的连续帧（second_in_video 差值 ≤ 1）构建窗口
      步长 stride=1，即相邻窗口重叠 T-1 帧
    """

    def __init__(
        self,
        csv_path: str = 'data/aligned_output/aligned_data.csv',
        window_size: int = 5,
        transform: Optional[object] = None,
        normalize_gnss: bool = True,
        gnss_stats_path: str = 'data/gnss_normalization.json',
        img_size: int = 224,
    ):
        """
        Args:
            csv_path: 对齐数据 CSV 文件路径（相对于项目根目录或绝对路径）
            window_size: 滑动窗口帧数 T（默认5）
            transform: 可选的图像变换（若 None 则使用默认的 Resize+Normalize）
            normalize_gnss: 是否对 GNSS 特征做 z-score 归一化
            gnss_stats_path: GNSS 归一化统计文件路径
            img_size: 图像缩放尺寸（默认224）
        """
        # ── 路径处理 ──────────────────────────────────────────────
        csv_path = Path(csv_path) if Path(csv_path).is_absolute() else PROJECT_ROOT / csv_path
        assert csv_path.exists(), f"CSV 文件不存在: {csv_path}"

        self.project_root = PROJECT_ROOT
        self.window_size = window_size
        self.normalize_gnss = normalize_gnss
        self.gnss_stats_path = PROJECT_ROOT / gnss_stats_path

        # ── 读取数据 ──────────────────────────────────────────────
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"[Dataset] 读取 {len(df)} 条记录，列: {list(df.columns)}")

        # 确认必要列存在
        required_cols = ['frame_path', 'video_file', 'second_in_video', '分类'] + GNSS_COLS
        missing = [c for c in required_cols if c not in df.columns]
        assert not missing, f"CSV 缺少列: {missing}"

        # 转换类型
        df['second_in_video'] = df['second_in_video'].astype(int)
        df['分类'] = df['分类'].astype(int)
        for col in GNSS_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)

        self.df = df.reset_index(drop=True)

        # ── 构建有效滑动窗口索引 ──────────────────────────────────
        self.windows = self._build_windows()
        print(f"[Dataset] 有效窗口数: {len(self.windows)}（window_size={window_size}）")

        # ── GNSS 归一化统计 ───────────────────────────────────────
        if normalize_gnss:
            self.gnss_mean, self.gnss_std = self._get_gnss_stats()

        # ── 图像变换 ──────────────────────────────────────────────
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

        # ── 计算并保存类别权重 ────────────────────────────────────
        self._save_class_weights()

    def _build_windows(self):
        """
        按 video_file 分组，提取所有连续 T 帧的有效窗口。
        连续性判断：相邻帧的 second_in_video 差值恰好为 1。

        Returns:
            windows: list of (起始行索引 in df, 中点帧行索引)
        """
        windows = []
        # 按视频文件分组处理
        for video_file, group in self.df.groupby('video_file', sort=False):
            group = group.sort_values('second_in_video').reset_index()
            # group['index'] 是原始 df 的行号
            idxs = group['index'].tolist()
            seconds = group['second_in_video'].tolist()

            n = len(idxs)
            for i in range(n - self.window_size + 1):
                # 检查时间连续性
                is_consecutive = all(
                    seconds[i + j + 1] - seconds[i + j] == 1
                    for j in range(self.window_size - 1)
                )
                if is_consecutive:
                    window_idxs = idxs[i: i + self.window_size]
                    mid_idx = idxs[i + self.window_size // 2]  # 中点帧
                    windows.append((window_idxs, mid_idx))
        return windows

    def _get_gnss_stats(self):
        """
        获取 GNSS 归一化统计（mean/std）。
        若统计文件不存在则从数据计算并保存。

        Returns:
            mean: np.ndarray (7,)
            std:  np.ndarray (7,)
        """
        if self.gnss_stats_path.exists():
            with open(self.gnss_stats_path) as f:
                stats = json.load(f)
            mean = np.array(stats['mean'], dtype=np.float32)
            std  = np.array(stats['std'], dtype=np.float32)
            print(f"[Dataset] 加载 GNSS 归一化统计: {self.gnss_stats_path}")
        else:
            gnss_data = self.df[GNSS_COLS].values.astype(np.float32)
            mean = gnss_data.mean(axis=0)
            std  = gnss_data.std(axis=0)
            std[std < 1e-6] = 1.0  # 避免除零（如常数列）

            stats = {'columns': GNSS_COLS, 'mean': mean.tolist(), 'std': std.tolist()}
            self.gnss_stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.gnss_stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            print(f"[Dataset] GNSS 归一化统计已保存: {self.gnss_stats_path}")
        return mean, std

    def _save_class_weights(self):
        """计算并保存类别权重（用于训练时缓解类别不平衡）"""
        weights_path = PROJECT_ROOT / 'data' / 'class_weights.json'
        if weights_path.exists():
            return

        # 统计窗口中点帧的类别分布
        labels = [self.df.iloc[mid]['分类'] for _, mid in self.windows]
        counts = np.bincount(labels, minlength=11).astype(float)
        counts[counts == 0] = 1.0  # 避免除零

        # 权重 = 1 / sqrt(count)，再归一化
        weights = 1.0 / np.sqrt(counts)
        weights = weights / weights.sum() * 11

        data = {
            'class_counts': counts.tolist(),
            'class_weights': weights.tolist(),
            'num_classes': 11
        }
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        with open(weights_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[Dataset] 类别权重已保存: {weights_path}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            video_tensor: (T, 3, 224, 224) — 归一化后的 RGB 帧序列
            gnss_tensor:  (7,)             — 归一化后的 GNSS 特征
            label:        int              — 类别标签 0~10
        """
        window_idxs, mid_idx = self.windows[idx]

        # ── 加载 T 帧图像 ─────────────────────────────────────────
        frames = []
        for row_idx in window_idxs:
            row = self.df.iloc[row_idx]
            img_path = self.project_root / row['frame_path']

            if not img_path.exists():
                warnings.warn(f"帧图片不存在: {img_path}，使用全零帧")
                # 用黑帧占位，保持维度一致
                img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            else:
                img = Image.open(img_path).convert('RGB')

            frames.append(self.transform(img))

        video_tensor = torch.stack(frames, dim=0)  # (T, 3, H, W)

        # ── 提取中点帧的 GNSS 特征 ────────────────────────────────
        mid_row = self.df.iloc[mid_idx]
        gnss = np.array([mid_row[c] for c in GNSS_COLS], dtype=np.float32)

        if self.normalize_gnss:
            gnss = (gnss - self.gnss_mean) / (self.gnss_std + 1e-6)

        gnss_tensor = torch.from_numpy(gnss)  # (7,)

        # ── 类别标签 ──────────────────────────────────────────────
        label = int(mid_row['分类'])
        assert 0 <= label <= 10, f"类别越界: {label}"

        return video_tensor, gnss_tensor, label


def split_dataset(
    dataset: AgriMultimodalDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    save_path: str = 'data/split_indices.json',
):
    """
    按比例分层划分数据集，保持各类别分布。

    Args:
        dataset: AgriMultimodalDataset 实例
        train_ratio / val_ratio / test_ratio: 划分比例（三者之和为1）
        seed: 随机种子
        save_path: 保存划分索引的 JSON 文件路径

    Returns:
        train_set, val_set, test_set: 三个 Subset
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    rng = np.random.default_rng(seed)

    # 提取所有样本的标签
    labels = np.array([
        int(dataset.df.iloc[mid]['分类'])
        for _, mid in dataset.windows
    ])

    train_idx, val_idx, test_idx = [], [], []

    # 按类别分层
    for cls in range(11):
        cls_indices = np.where(labels == cls)[0]
        if len(cls_indices) == 0:
            continue
        rng.shuffle(cls_indices)

        n = len(cls_indices)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        train_idx.extend(cls_indices[:n_train].tolist())
        val_idx.extend(cls_indices[n_train: n_train + n_val].tolist())
        test_idx.extend(cls_indices[n_train + n_val:].tolist())

    # 保存划分索引
    save_path = PROJECT_ROOT / save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump({'train': train_idx, 'val': val_idx, 'test': test_idx, 'seed': seed}, f)
    print(f"[Split] train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


# ── 直接运行时做快速验证 ────────────────────────────────────────────────────
if __name__ == '__main__':
    import time

    print("=" * 60)
    print("AgriMultimodalDataset 快速验证")
    print("=" * 60)

    dataset = AgriMultimodalDataset(
        csv_path='data/aligned_output/aligned_data.csv',
        window_size=5,
        normalize_gnss=True,
    )

    print(f"\n数据集大小: {len(dataset)} 个窗口")

    # 取一个样本查看形状
    video, gnss, label = dataset[0]
    print(f"video_tensor 形状: {tuple(video.shape)}  (期望: (5, 3, 224, 224))")
    print(f"gnss_tensor  形状: {tuple(gnss.shape)}   (期望: (7,))")
    print(f"label:        {label}  (期望: 0~10)")
    assert video.shape == (5, 3, 224, 224), f"video 形状错误: {video.shape}"
    assert gnss.shape == (7,), f"gnss 形状错误: {gnss.shape}"
    assert 0 <= label <= 10, f"label 越界: {label}"
    assert not torch.isnan(video).any(), "video 中存在 NaN"
    assert not torch.isnan(gnss).any(), "gnss 中存在 NaN"

    # DataLoader 测试
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    t0 = time.time()
    v_batch, g_batch, l_batch = next(iter(loader))
    elapsed = time.time() - t0
    print(f"\nDataLoader 批次形状:")
    print(f"  video: {tuple(v_batch.shape)}")
    print(f"  gnss:  {tuple(g_batch.shape)}")
    print(f"  label: {tuple(l_batch.shape)}, 值范围: [{l_batch.min()}, {l_batch.max()}]")
    print(f"  加载耗时: {elapsed*1000:.1f} ms")

    print("\n✅ 所有验证通过")
