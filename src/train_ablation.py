#!/usr/bin/env python3
"""
Ablation training for trajectory-only, image-only, and multimodal models.

Designed for long trajectory windows, e.g. seq_len=1000. Image and multimodal
experiments sample a small number of frames from the same long window to keep
ViT memory practical.
"""

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-agri-mbt")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from dataset import CLASS_NAMES, ENGINEERED_FEATURE_NAMES, GNSS_COLS, IMAGENET_MEAN, IMAGENET_STD, PROJECT_ROOT
from models.visual_encoder import VisualEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Ablation training")
    parser.add_argument("--mode", choices=["trajectory_only", "image_only", "multimodal"], required=True)
    parser.add_argument("--train-csv", default="data/taif_20241018_split/train.csv")
    parser.add_argument("--val-csv", default="data/taif_20241018_split/val.csv")
    parser.add_argument("--test-csv", default="data/taif_20241018_split/test.csv")
    parser.add_argument("--save-dir", default="experiments/ablation")
    parser.add_argument("--seq-len", type=int, default=512, help="Maximum trajectory sequence length")
    parser.add_argument("--stride", type=int, default=20, help="Training stride for fixed sampling")
    parser.add_argument("--eval-stride", type=int, default=1, help="Validation/test stride; 1 means dense per-point evaluation")
    parser.add_argument("--context-mode", choices=["causal", "centered"], default="causal")
    parser.add_argument("--sampling-strategy", choices=["fixed", "adaptive"], default="fixed")
    parser.add_argument("--duration-stats", default="", help="duration_sampling_config.json from analyze_behavior_durations.py")
    parser.add_argument("--adaptive-min-window", type=int, default=16)
    parser.add_argument("--adaptive-max-window", type=int, default=512)
    parser.add_argument("--adaptive-context-scale", type=float, default=2.0)
    parser.add_argument("--adaptive-min-stride", type=int, default=1)
    parser.add_argument("--adaptive-max-stride", type=int, default=20)
    parser.add_argument("--adaptive-stride-ratio", type=float, default=0.25)
    parser.add_argument("--image-window-size", type=int, default=5)
    parser.add_argument("--image-sampling", choices=["center", "uniform"], default="center")
    parser.add_argument("--image-radius", type=int, default=8, help="Center sampling radius in seconds")
    parser.add_argument("--image-radius-mode", choices=["fixed", "duration"], default="fixed")
    parser.add_argument("--image-radius-duration-scale", type=float, default=0.5)
    parser.add_argument("--image-radius-classes", default="", help="Comma-separated class ids that use duration-based image radius; empty means all classes")
    parser.add_argument("--image-temporal-pool", choices=["mean", "transformer", "gru"], default="mean")
    parser.add_argument("--image-temporal-delta", choices=["none", "diff"], default="none")
    parser.add_argument("--feature-mode", choices=["raw", "engineered"], default="engineered")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight-power", type=float, default=0.5, help="Use inverse class frequency raised to this power")
    parser.add_argument("--train-sampler", choices=["shuffle", "class_balanced"], default="shuffle")
    parser.add_argument("--sampler-weight-power", type=float, default=0.5)
    parser.add_argument("--aux-target-classes", default="", help="Comma-separated class ids for one-vs-rest auxiliary heads")
    parser.add_argument("--aux-loss-weight", type=float, default=0.0)
    parser.add_argument("--aux-pos-weight-power", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--rnn-layers", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--all-gpus", action="store_true")
    parser.add_argument("--gpu-ids", default="")
    parser.add_argument("--fusion", choices=["concat", "mbt"], default="concat")
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--fusion-heads", type=int, default=8)
    parser.add_argument("--num-latents", type=int, default=4)
    parser.add_argument("--freeze-encoders-epochs", type=int, default=0)
    parser.add_argument("--init-traj-checkpoint", default="")
    parser.add_argument("--init-image-checkpoint", default="")
    parser.add_argument("--eval-checkpoint", default="", help="Evaluate an existing checkpoint and write summary artifacts without training")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--early-stop-val-macro-f1", type=float, default=0.0)
    parser.add_argument("--quiet-warnings", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def configure_warnings(quiet: bool):
    if not quiet:
        return
    warnings.filterwarnings("ignore", message=".*imbalance between your GPUs.*")
    warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")
    warnings.filterwarnings("ignore", message=".*Was asked to gather along dimension 0.*")
    warnings.filterwarnings("ignore", message=".*Attempting to run cuBLAS.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.parallel.parallel_apply")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.parallel")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.linear")


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_class_ids(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


class LongWindowDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        seq_len: int,
        stride: int,
        eval_stride: int,
        context_mode: str,
        sampling_strategy: str,
        duration_stats: str,
        adaptive_min_window: int,
        adaptive_max_window: int,
        adaptive_context_scale: float,
        adaptive_min_stride: int,
        adaptive_max_stride: int,
        adaptive_stride_ratio: float,
        image_window_size: int,
        image_sampling: str,
        image_radius: int,
        image_radius_mode: str,
        image_radius_duration_scale: float,
        image_radius_classes: str,
        feature_mode: str,
        mode: str,
        is_train: bool,
        transform=None,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        img_size: int = 224,
    ):
        path = Path(csv_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        self.df = pd.read_csv(path, encoding="utf-8-sig")
        self.df["frame_time"] = pd.to_datetime(self.df["frame_time"])
        self.df["second_in_video"] = self.df["second_in_video"].astype(int)
        self.df["分类"] = self.df["分类"].astype(int)
        for col in GNSS_COLS:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0.0).astype(float)

        self.seq_len = seq_len
        self.stride = stride
        self.eval_stride = eval_stride if eval_stride > 0 else stride
        self.context_mode = context_mode
        self.sampling_strategy = sampling_strategy
        self.is_train = is_train
        self.adaptive_min_window = adaptive_min_window
        self.adaptive_max_window = adaptive_max_window
        self.adaptive_context_scale = adaptive_context_scale
        self.adaptive_min_stride = adaptive_min_stride
        self.adaptive_max_stride = adaptive_max_stride
        self.adaptive_stride_ratio = adaptive_stride_ratio
        self.image_window_size = image_window_size
        self.image_sampling = image_sampling
        self.image_radius = image_radius
        self.image_radius_mode = image_radius_mode
        self.image_radius_duration_scale = image_radius_duration_scale
        self.image_radius_classes = {
            int(x.strip()) for x in image_radius_classes.split(",") if x.strip()
        }
        self.feature_mode = feature_mode
        self.mode = mode
        self.feature_names = GNSS_COLS if feature_mode == "raw" else ENGINEERED_FEATURE_NAMES
        self.feature_dim = len(self.feature_names)
        self.features_all = self._precompute_features()
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)
        self.project_root = PROJECT_ROOT
        self.duration_config = self._load_duration_config(duration_stats)
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.windows = self._build_windows()
        print(
            f"[AblationDataset] {path} rows={len(self.df)} windows={len(self.windows)} "
            f"mode={mode} seq_len={seq_len} context={context_mode} "
            f"sampling={sampling_strategy if is_train else 'eval_dense'} image_sampling={image_sampling} "
            f"image_radius={image_radius_mode}:{image_radius}"
        )

    def _load_duration_config(self, duration_stats: str) -> dict:
        if not duration_stats:
            return {}
        path = Path(duration_stats)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"duration stats not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    def _infer_adaptive_stride(self, group: pd.DataFrame, start: int, end: int) -> dict:
        labels = group["分类"].to_numpy()
        seconds = group["second_in_video"].to_numpy()
        durations = {i: [] for i in range(11)}
        run_start = start
        for pos in range(start + 1, end + 1):
            if labels[pos] != labels[pos - 1]:
                durations[int(labels[run_start])].append(int(seconds[pos - 1] - seconds[run_start] + 1))
                run_start = pos
        durations[int(labels[run_start])].append(int(seconds[end] - seconds[run_start] + 1))
        strides = {}
        for class_id, values in durations.items():
            if not values:
                continue
            median = max(float(np.median(values)), 1.0)
            strides[class_id] = int(np.clip(np.ceil(median * self.adaptive_stride_ratio), self.adaptive_min_stride, self.adaptive_max_stride))
        return strides

    def _class_stride(self, class_id: int, inferred: dict) -> int:
        if class_id in self.duration_config:
            return int(np.clip(self.duration_config[class_id].get("stride", self.stride), self.adaptive_min_stride, self.adaptive_max_stride))
        if class_id in inferred:
            return int(inferred[class_id])
        return int(self.stride)

    def _iter_contiguous_segments(self, seconds: np.ndarray):
        start = 0
        for pos in range(1, len(seconds)):
            if seconds[pos] - seconds[pos - 1] > 1:
                yield start, pos - 1
                start = pos
        if len(seconds) > 0:
            yield start, len(seconds) - 1

    def _adaptive_centers(self, group: pd.DataFrame, seg_start: int, seg_end: int) -> list[int]:
        labels = group["分类"].to_numpy()
        inferred = self._infer_adaptive_stride(group, seg_start, seg_end)
        centers = []
        run_start = seg_start
        for pos in range(seg_start + 1, seg_end + 1):
            if labels[pos] != labels[pos - 1]:
                stride = self._class_stride(int(labels[run_start]), inferred)
                centers.extend(range(run_start, pos, stride))
                if centers[-1] != pos - 1:
                    centers.append(pos - 1)
                run_start = pos
        stride = self._class_stride(int(labels[run_start]), inferred)
        centers.extend(range(run_start, seg_end + 1, stride))
        if centers[-1] != seg_end:
            centers.append(seg_end)
        return sorted(set(int(c) for c in centers))

    def _fixed_centers(self, seg_start: int, seg_end: int, stride: int) -> list[int]:
        stride = max(int(stride), 1)
        centers = list(range(seg_start, seg_end + 1, stride))
        if not centers or centers[-1] != seg_end:
            centers.append(seg_end)
        return centers

    def _window_indices(self, idxs: np.ndarray, seg_start: int, seg_end: int, center_pos: int):
        if self.context_mode == "causal":
            positions = np.arange(center_pos - self.seq_len + 1, center_pos + 1)
            anchor_slot = self.seq_len - 1
        else:
            half = self.seq_len // 2
            positions = np.arange(center_pos - half, center_pos - half + self.seq_len)
            anchor_slot = min(half, self.seq_len - 1)
        positions = np.clip(positions, seg_start, seg_end)
        return idxs[positions].tolist(), int(idxs[center_pos]), int(anchor_slot)

    def _build_windows(self):
        windows = []
        for _, group in self.df.groupby("video_file", sort=False):
            group = group.sort_values(["second_in_video", "frame_time"]).reset_index()
            idxs = group["index"].to_numpy()
            seconds = group["second_in_video"].to_numpy()
            if len(idxs) == 0:
                continue
            for seg_start, seg_end in self._iter_contiguous_segments(seconds):
                if self.is_train and self.sampling_strategy == "adaptive":
                    centers = self._adaptive_centers(group, seg_start, seg_end)
                else:
                    stride = self.stride if self.is_train else self.eval_stride
                    centers = self._fixed_centers(seg_start, seg_end, stride)
                for center_pos in centers:
                    windows.append(self._window_indices(idxs, seg_start, seg_end, center_pos))
        return windows

    def _circular_diff(self, values: np.ndarray) -> np.ndarray:
        diff = np.diff(values, prepend=values[:1])
        return ((diff + 180.0) % 360.0) - 180.0

    def _rolling_stat(self, values: np.ndarray, window: int, stat: str) -> np.ndarray:
        out = np.zeros_like(values, dtype=np.float32)
        for i in range(len(values)):
            start = max(0, i - window + 1)
            segment = values[start:i + 1]
            if stat == "median":
                out[i] = np.median(segment)
            elif stat == "std":
                out[i] = np.std(segment) if len(segment) > 1 else 0.0
            else:
                raise ValueError(stat)
        return out

    def _make_features(self, raw: np.ndarray) -> np.ndarray:
        if self.feature_mode == "raw":
            return raw
        lon, lat, speed, depth, heading = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3], raw[:, 4]
        dx = np.diff(lon, prepend=lon[:1])
        dy = np.diff(lat, prepend=lat[:1])
        acceleration = np.diff(speed, prepend=speed[:1])
        angle_diff = self._circular_diff(heading)
        angular_speed = angle_diff
        angular_acceleration = np.diff(angular_speed, prepend=angular_speed[:1])
        jerk = np.diff(acceleration, prepend=acceleration[:1])
        curvature = angular_speed / (np.abs(speed) + 1e-3)
        motion_energy = speed ** 2 + np.abs(acceleration)
        turn_indicator = np.abs(angle_diff)
        depth_delta = np.diff(depth, prepend=depth[:1])
        local_speed_mean = np.array([speed[max(0, i - 2): i + 1].mean() for i in range(len(speed))], dtype=np.float32)
        local_heading_std = np.array([heading[max(0, i - 2): i + 1].std() for i in range(len(heading))], dtype=np.float32)
        parts = [lon, lat, speed, acceleration, angular_speed, angular_acceleration, angle_diff]
        for series in [speed, acceleration, angular_speed, angular_acceleration, angle_diff]:
            parts.extend([
                self._rolling_stat(series, 5, "median"),
                self._rolling_stat(series, 20, "median"),
                self._rolling_stat(series, 5, "std"),
                self._rolling_stat(series, 20, "std"),
            ])
        parts.extend([dx, dy, jerk, curvature, motion_energy, turn_indicator, local_speed_mean, local_heading_std, depth_delta])
        return np.stack(parts, axis=-1).astype(np.float32)

    def _precompute_features(self) -> np.ndarray:
        features = np.zeros((len(self.df), self.feature_dim), dtype=np.float32)
        for _, group in self.df.groupby("video_file", sort=False):
            group = group.sort_values(["second_in_video", "frame_time"])
            raw = group[GNSS_COLS].values.astype(np.float32)
            features[group.index.to_numpy()] = self._make_features(raw)
        return features

    def _extract_features(self, indices) -> np.ndarray:
        return self.features_all[np.asarray(indices, dtype=np.int64)]

    def compute_stats(self):
        chunks = [self._extract_features(indices) for indices, _, _ in self.windows]
        data = np.concatenate(chunks, axis=0).astype(np.float32)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std[std < 1e-6] = 1.0
        return mean, std

    def set_stats(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.std[self.std < 1e-6] = 1.0

    def __len__(self):
        return len(self.windows)

    def _load_images(self, indices, anchor_slot: int):
        center_pos = int(anchor_slot)
        if self.image_window_size <= 1:
            positions = [center_pos]
        elif self.image_sampling == "uniform":
            positions = np.linspace(0, len(indices) - 1, self.image_window_size).round().astype(int).tolist()
        else:
            center_second = int(self.df.iloc[indices[center_pos]]["second_in_video"])
            class_id = int(self.df.iloc[indices[center_pos]]["分类"])
            radius = self._image_radius_for_class(class_id)
            target_offsets = np.linspace(-radius, radius, self.image_window_size)
            seconds = self.df.iloc[indices]["second_in_video"].astype(int).to_numpy()
            positions = []
            for offset in target_offsets:
                target_second = center_second + int(round(offset))
                positions.append(int(np.argmin(np.abs(seconds - target_second))))
        frames = []
        for pos in positions:
            row = self.df.iloc[indices[pos]]
            img_path = self.project_root / row["frame_path"]
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
            else:
                img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            frames.append(self.transform(img))
        return torch.stack(frames, dim=0)

    def _image_radius_for_class(self, class_id: int) -> int:
        if self.image_radius_mode != "duration":
            return int(self.image_radius)
        if self.image_radius_classes and class_id not in self.image_radius_classes:
            return int(self.image_radius)
        cfg = self.duration_config.get(class_id, {})
        duration = float(cfg.get("p75_duration", cfg.get("median_duration", self.image_radius * 2)))
        radius = int(round(duration * self.image_radius_duration_scale))
        return int(np.clip(radius, 1, max(self.image_radius * 4, 1)))

    def __getitem__(self, idx):
        indices, center_idx, anchor_slot = self.windows[idx]
        features = self._extract_features(indices)
        if self.mean is not None and self.std is not None:
            features = (features - self.mean) / (self.std + 1e-6)
        traj = torch.from_numpy(features.astype(np.float32))
        if self.mode == "trajectory_only":
            images = torch.empty(0)
        else:
            images = self._load_images(indices, anchor_slot)
        label = int(self.df.iloc[center_idx]["分类"])
        return traj, images, label


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.score(torch.tanh(x)).squeeze(-1), dim=1)
        return (x * weights.unsqueeze(-1)).sum(dim=1)


class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_size=256, layers=2, dropout=0.35):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm(input_dim)
        self.rnn = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )
        self.pool = AttentionPool(hidden_size)

    def forward_tokens(self, x):
        x = self.norm(x)
        out, _ = self.rnn(x)
        out = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]
        return self.proj(out)

    def forward(self, x):
        return self.pool(self.forward_tokens(x))


class ImageEncoder(nn.Module):
    def __init__(self, pretrained=False, dropout=0.35, temporal_pool="mean", temporal_delta="none", max_frames=32):
        super().__init__()
        self.visual = VisualEncoder(pretrained=pretrained)
        self.temporal_pool = temporal_pool
        self.temporal_delta = temporal_delta
        self.frame_pool = nn.Sequential(nn.Linear(768, 768), nn.GELU(), nn.Linear(768, 1))
        if temporal_delta == "diff":
            self.delta_proj = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, 768), nn.GELU())
        if temporal_pool == "transformer":
            self.frame_pos = nn.Parameter(torch.zeros(1, max_frames, 768))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=768,
                nhead=8,
                dim_feedforward=1536,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        elif temporal_pool == "gru":
            self.temporal_encoder = nn.GRU(768, 384, batch_first=True, bidirectional=True)
        if temporal_pool in {"transformer", "gru"}:
            self.temporal_score = nn.Linear(768, 1)
        self.out = nn.Sequential(nn.LayerNorm(768), nn.Dropout(dropout))

    def forward_tokens(self, images):
        tokens = self.visual(images)
        b, t, p, d = tokens.shape
        flat = tokens.reshape(b * t, p, d)
        weights = torch.softmax(self.frame_pool(flat), dim=1)
        return (flat * weights).sum(dim=1).reshape(b, t, d)

    def forward(self, images):
        frame_feats = self.forward_tokens(images)
        if self.temporal_delta == "diff":
            delta = frame_feats - torch.cat([frame_feats[:, :1], frame_feats[:, :-1]], dim=1)
            frame_feats = self.delta_proj(torch.cat([frame_feats, delta], dim=-1))
        if self.temporal_pool == "transformer":
            if frame_feats.size(1) > self.frame_pos.size(1):
                raise ValueError(f"image window has {frame_feats.size(1)} frames, max supported is {self.frame_pos.size(1)}")
            encoded = self.temporal_encoder(frame_feats + self.frame_pos[:, :frame_feats.size(1)])
            weights = torch.softmax(self.temporal_score(torch.tanh(encoded)).squeeze(-1), dim=1)
            return self.out((encoded * weights.unsqueeze(-1)).sum(dim=1))
        if self.temporal_pool == "gru":
            encoded, _ = self.temporal_encoder(frame_feats)
            weights = torch.softmax(self.temporal_score(torch.tanh(encoded)).squeeze(-1), dim=1)
            return self.out((encoded * weights.unsqueeze(-1)).sum(dim=1))
        return self.out(frame_feats.mean(dim=1))


class BottleneckFusionBlock(nn.Module):
    def __init__(self, dim=768, heads=8, dropout=0.1):
        super().__init__()
        self.latent_from_modal = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.traj_from_latent = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.image_from_latent = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_latent = nn.LayerNorm(dim)
        self.norm_traj = nn.LayerNorm(dim)
        self.norm_image = nn.LayerNorm(dim)
        self.ffn_latent = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.ffn_traj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.ffn_image = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.scale_latent = nn.Parameter(torch.tensor(1e-3))
        self.scale_traj = nn.Parameter(torch.zeros(1))
        self.scale_image = nn.Parameter(torch.zeros(1))
        self.scale_latent_ffn = nn.Parameter(torch.tensor(1e-3))
        self.scale_traj_ffn = nn.Parameter(torch.tensor(1e-3))
        self.scale_image_ffn = nn.Parameter(torch.tensor(1e-3))

    def forward(self, traj_tokens, image_tokens, latents):
        modal_tokens = torch.cat([traj_tokens, image_tokens], dim=1)
        latent_update, _ = self.latent_from_modal(self.norm_latent(latents), modal_tokens, modal_tokens, need_weights=False)
        latents = latents + self.scale_latent * latent_update
        traj_update, _ = self.traj_from_latent(self.norm_traj(traj_tokens), latents, latents, need_weights=False)
        image_update, _ = self.image_from_latent(self.norm_image(image_tokens), latents, latents, need_weights=False)
        traj_tokens = traj_tokens + self.scale_traj * traj_update
        image_tokens = image_tokens + self.scale_image * image_update
        latents = latents + self.scale_latent_ffn * self.ffn_latent(latents)
        traj_tokens = traj_tokens + self.scale_traj_ffn * self.ffn_traj(traj_tokens)
        image_tokens = image_tokens + self.scale_image_ffn * self.ffn_image(image_tokens)
        return traj_tokens, image_tokens, latents


class MBTLikeFusion(nn.Module):
    def __init__(self, traj_dim, image_dim=768, fusion_dim=768, layers=2, heads=8, num_latents=4, dropout=0.1):
        super().__init__()
        self.traj_proj = nn.Linear(traj_dim, fusion_dim)
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.latents = nn.Parameter(torch.empty(1, num_latents, fusion_dim).normal_(std=0.02))
        self.blocks = nn.ModuleList([
            BottleneckFusionBlock(fusion_dim, heads=heads, dropout=dropout)
            for _ in range(layers)
        ])
        self.traj_pool = AttentionPool(fusion_dim)
        self.image_pool = AttentionPool(fusion_dim)
        self.out = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Dropout(dropout))

    def forward(self, traj_tokens, image_tokens):
        traj_tokens = self.traj_proj(traj_tokens)
        image_tokens = self.image_proj(image_tokens)
        latents = self.latents.expand(traj_tokens.size(0), -1, -1)
        for block in self.blocks:
            traj_tokens, image_tokens, latents = block(traj_tokens, image_tokens, latents)
        fused = (self.traj_pool(traj_tokens) + self.image_pool(image_tokens) + latents.mean(dim=1)) / 3.0
        return self.out(fused)


class AblationModel(nn.Module):
    def __init__(self, mode, traj_dim, hidden_size, layers, dropout, pretrained, fusion, fusion_layers, fusion_heads, num_latents, image_temporal_pool, image_temporal_delta, image_window_size, aux_target_classes):
        super().__init__()
        self.mode = mode
        self.fusion_mode = fusion
        self.aux_target_classes = list(aux_target_classes)
        if mode in {"trajectory_only", "multimodal"}:
            self.traj_encoder = TrajectoryEncoder(traj_dim, hidden_size, layers, dropout)
        if mode in {"image_only", "multimodal"}:
            self.image_encoder = ImageEncoder(
                pretrained=pretrained,
                dropout=dropout,
                temporal_pool=image_temporal_pool,
                temporal_delta=image_temporal_delta,
                max_frames=max(image_window_size, 32),
            )
        if mode == "trajectory_only":
            in_dim = hidden_size
        elif mode == "image_only":
            in_dim = 768
        elif fusion == "mbt":
            self.mbt_fusion = MBTLikeFusion(
                traj_dim=hidden_size,
                image_dim=768,
                fusion_dim=768,
                layers=fusion_layers,
                heads=fusion_heads,
                num_latents=num_latents,
                dropout=dropout,
            )
            in_dim = 768
        else:
            in_dim = hidden_size + 768
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 11),
        )
        self.aux_classifier = nn.Linear(in_dim, len(self.aux_target_classes)) if self.aux_target_classes else None

    def _emit_logits(self, features):
        logits = self.classifier(features)
        if self.aux_classifier is None:
            return logits
        return logits, self.aux_classifier(features)

    def forward(self, traj, images):
        if self.mode == "multimodal" and self.fusion_mode == "mbt":
            traj_tokens = self.traj_encoder.forward_tokens(traj)
            image_tokens = self.image_encoder.forward_tokens(images)
            return self._emit_logits(self.mbt_fusion(traj_tokens, image_tokens))
        feats = []
        if self.mode in {"trajectory_only", "multimodal"}:
            feats.append(self.traj_encoder(traj))
        if self.mode in {"image_only", "multimodal"}:
            feats.append(self.image_encoder(images))
        return self._emit_logits(torch.cat(feats, dim=1) if len(feats) > 1 else feats[0])


def build_loaders(args):
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)) if hasattr(args, "img_size") else transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    train_ds = LongWindowDataset(
        args.train_csv,
        args.seq_len,
        args.stride,
        args.eval_stride,
        args.context_mode,
        args.sampling_strategy,
        args.duration_stats,
        args.adaptive_min_window,
        args.adaptive_max_window,
        args.adaptive_context_scale,
        args.adaptive_min_stride,
        args.adaptive_max_stride,
        args.adaptive_stride_ratio,
        args.image_window_size,
        args.image_sampling,
        args.image_radius,
        args.image_radius_mode,
        args.image_radius_duration_scale,
        args.image_radius_classes,
        args.feature_mode,
        args.mode,
        True,
        transform,
    )
    mean, std = train_ds.compute_stats()
    train_ds.set_stats(mean, std)
    val_ds = LongWindowDataset(
        args.val_csv,
        args.seq_len,
        args.stride,
        args.eval_stride,
        args.context_mode,
        args.sampling_strategy,
        args.duration_stats,
        args.adaptive_min_window,
        args.adaptive_max_window,
        args.adaptive_context_scale,
        args.adaptive_min_stride,
        args.adaptive_max_stride,
        args.adaptive_stride_ratio,
        args.image_window_size,
        args.image_sampling,
        args.image_radius,
        args.image_radius_mode,
        args.image_radius_duration_scale,
        args.image_radius_classes,
        args.feature_mode,
        args.mode,
        False,
        transform,
        mean,
        std,
    )
    test_ds = LongWindowDataset(
        args.test_csv,
        args.seq_len,
        args.stride,
        args.eval_stride,
        args.context_mode,
        args.sampling_strategy,
        args.duration_stats,
        args.adaptive_min_window,
        args.adaptive_max_window,
        args.adaptive_context_scale,
        args.adaptive_min_stride,
        args.adaptive_max_stride,
        args.adaptive_stride_ratio,
        args.image_window_size,
        args.image_sampling,
        args.image_radius,
        args.image_radius_mode,
        args.image_radius_duration_scale,
        args.image_radius_classes,
        args.feature_mode,
        args.mode,
        False,
        transform,
        mean,
        std,
    )
    kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    train_sampler = None
    train_shuffle = True
    if args.train_sampler == "class_balanced":
        labels = torch.tensor([int(train_ds.df.iloc[center]["分类"]) for _, center, _ in train_ds.windows], dtype=torch.long)
        counts = torch.bincount(labels, minlength=11).float().clamp(min=1.0)
        sample_weights = counts[labels].pow(-float(args.sampler_weight_power))
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_shuffle = False
    return (
        train_ds,
        DataLoader(train_ds, shuffle=train_shuffle, sampler=train_sampler, **kwargs),
        DataLoader(val_ds, shuffle=False, **kwargs),
        DataLoader(test_ds, shuffle=False, **kwargs),
        mean,
        std,
    )


def class_weights(dataset, device, power: float):
    labels = [int(dataset.df.iloc[center]["分类"]) for _, center, _ in dataset.windows]
    counts = torch.bincount(torch.tensor(labels), minlength=11).float().clamp(min=1.0)
    weights = counts.pow(-float(power))
    return (weights / weights.sum() * 11).to(device)


def aux_pos_weights(dataset, target_classes: list[int], device, power: float):
    if not target_classes:
        return None
    labels = torch.tensor([int(dataset.df.iloc[center]["分类"]) for _, center, _ in dataset.windows], dtype=torch.long)
    weights = []
    total = float(len(labels))
    for class_id in target_classes:
        pos = float((labels == int(class_id)).sum().item())
        neg = max(total - pos, 1.0)
        pos = max(pos, 1.0)
        weights.append((neg / pos) ** float(power))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def set_multimodal_encoder_trainable(model, trainable: bool):
    base = unwrap_model(model)
    for name in ["traj_encoder", "image_encoder"]:
        module = getattr(base, name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = trainable


def load_branch_checkpoint(model, checkpoint_path: str, branch_prefix: str, device):
    if not checkpoint_path:
        return
    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    source_state = checkpoint.get("model_state", checkpoint)
    target = unwrap_model(model)
    target_state = target.state_dict()
    matched = {}
    skipped = 0
    prefix = f"{branch_prefix}."
    for key, value in source_state.items():
        if not key.startswith(prefix):
            continue
        if key in target_state and target_state[key].shape == value.shape:
            matched[key] = value
        else:
            skipped += 1
    if not matched:
        raise RuntimeError(f"no compatible {branch_prefix} weights found in {path}")
    target_state.update(matched)
    target.load_state_dict(target_state)
    print(f"[Ablation] Loaded {len(matched)} {branch_prefix} tensors from {path} (skipped {skipped})")


def load_unimodal_initialization(model, args, device):
    if args.mode != "multimodal":
        return
    load_branch_checkpoint(model, args.init_traj_checkpoint, "traj_encoder", device)
    load_branch_checkpoint(model, args.init_image_checkpoint, "image_encoder", device)


def unpack_model_output(output):
    if isinstance(output, tuple):
        return output
    return output, None


def run_epoch(model, loader, optimizer, criterion, aux_criterion, device, train, args):
    model.train(train)
    total_loss = total_correct = total_samples = 0
    preds_all, labels_all = [], []
    max_batches = args.max_train_batches if train else args.max_eval_batches
    for batch_idx, (traj, images, labels) in enumerate(loader, start=1):
        traj = traj.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        images = images.to(device, non_blocking=True) if args.mode != "trajectory_only" else images
        with torch.set_grad_enabled(train):
            logits, aux_logits = unpack_model_output(model(traj, images))
            loss = criterion(logits, labels)
            if aux_criterion is not None and aux_logits is not None:
                target_classes = torch.tensor(args.aux_target_class_ids, dtype=torch.long, device=device)
                aux_targets = (labels.unsqueeze(1) == target_classes.unsqueeze(0)).float()
                loss = loss + float(args.aux_loss_weight) * aux_criterion(aux_logits, aux_targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at batch {batch_idx}: {loss.item()}")
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        preds_all.append(preds.detach().cpu())
        labels_all.append(labels.detach().cpu())
        if max_batches and batch_idx >= max_batches:
            break
    labels_np = torch.cat(labels_all).numpy() if labels_all else np.array([])
    preds_np = torch.cat(preds_all).numpy() if preds_all else np.array([])
    return {
        "loss": total_loss / max(total_samples, 1),
        "acc": total_correct / max(total_samples, 1),
        "macro_f1": f1_score(labels_np, preds_np, average="macro", zero_division=0) if len(labels_np) else 0.0,
        "weighted_f1": f1_score(labels_np, preds_np, average="weighted", zero_division=0) if len(labels_np) else 0.0,
        "per_class_recall": recall_score(labels_np, preds_np, average=None, labels=list(range(11)), zero_division=0).tolist() if len(labels_np) else [0.0] * 11,
        "confusion_matrix": confusion_matrix(labels_np, preds_np, labels=list(range(11))).tolist() if len(labels_np) else [[0] * 11 for _ in range(11)],
    }


def collect_predictions(model, loader, device, args):
    model.eval()
    rows = []
    offset = 0
    with torch.no_grad():
        for traj, images, labels in loader:
            traj = traj.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images = images.to(device, non_blocking=True) if args.mode != "trajectory_only" else images
            logits, _ = unpack_model_output(model(traj, images))
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            for i in range(len(labels_np)):
                rows.append({
                    "dataset_index": offset + i,
                    "y_true": int(labels_np[i]),
                    "y_pred": int(preds[i]),
                    "correct": bool(labels_np[i] == preds[i]),
                })
            offset += len(labels_np)
    return pd.DataFrame(rows)


def save_plots(save_dir: Path, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for item in summary["history"]:
        row = {"epoch": item["epoch"]}
        for split in ["train", "val"]:
            for metric in ["loss", "acc", "macro_f1", "weighted_f1"]:
                row[f"{split}_{metric}"] = item[split][metric]
        rows.append(row)
    pd.DataFrame(rows).to_csv(save_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    epochs = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=160)
    for ax, metric, title in zip(axes, ["loss", "acc", "macro_f1"], ["Loss", "Accuracy", "Macro F1"]):
        ax.plot(epochs, [r[f"train_{metric}"] for r in rows], label="train")
        ax.plot(epochs, [r[f"val_{metric}"] for r in rows], label="val")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "training_curves.png")
    plt.close(fig)

    mat = np.asarray(summary["test"]["confusion_matrix"], dtype=np.float32)
    norm = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1.0)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Test Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(11))
    ax.set_yticks(range(11))
    for i in range(11):
        for j in range(11):
            if mat[i, j] > 0:
                ax.text(j, i, str(int(mat[i, j])), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_dir / "confusion_matrix.png")
    plt.close(fig)

    per_class_rows = []
    for class_id in range(11):
        tp = mat[class_id, class_id]
        support = mat[class_id, :].sum()
        predicted = mat[:, class_id].sum()
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class_rows.append({
            "class_id": class_id,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support),
        })
    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df.to_csv(save_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=160)
    x = np.arange(11)
    width = 0.25
    ax.bar(x - width, per_class_df["precision"], width, label="Precision")
    ax.bar(x, per_class_df["recall"], width, label="Recall")
    ax.bar(x + width, per_class_df["f1"], width, label="F1")
    ax.set_ylim(0, 1)
    ax.set_title("Test Per-Class Metrics")
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Score")
    ax.set_xticks(range(11))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "per_class_metrics.png")
    plt.close(fig)


def save_spatial_errors(save_dir: Path, predictions_df: pd.DataFrame, mode: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not {"经度", "纬度", "correct"}.issubset(predictions_df.columns):
        return
    correct = predictions_df["correct"].astype(bool)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    ax.scatter(predictions_df.loc[correct, "经度"], predictions_df.loc[correct, "纬度"],
               c="green", s=8, alpha=0.35, label="Correct", linewidths=0)
    ax.scatter(predictions_df.loc[~correct, "经度"], predictions_df.loc[~correct, "纬度"],
               c="red", s=14, alpha=0.65, marker="x", label="Wrong")
    acc = correct.mean() * 100 if len(correct) else 0.0
    ax.set_title(f"{mode} Spatial Prediction Errors (Acc: {acc:.2f}%)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "spatial_errors.png")
    plt.close(fig)


def write_evaluation_artifacts(save_dir: Path, args, model, test_loader, criterion, aux_criterion, device, best_val: float, history: list):
    test_metrics = run_epoch(model, test_loader, None, criterion, aux_criterion, device, False, args)
    predictions = collect_predictions(model, test_loader, device, args)
    test_meta = test_loader.dataset.df.iloc[
        [center for _, center, _ in test_loader.dataset.windows]
    ].reset_index(drop=True)
    predictions = pd.concat([test_meta, predictions], axis=1)
    predictions.to_csv(save_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    summary = {
        "mode": args.mode,
        "best_val_macro_f1": best_val,
        "test": test_metrics,
        "history": history,
        "class_names": CLASS_NAMES,
        "args": vars(args),
    }
    with open(save_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(save_dir / "confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump({"class_names": CLASS_NAMES, "matrix": test_metrics["confusion_matrix"]}, f, ensure_ascii=False, indent=2)
    with open(save_dir / "per_class_recall.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics["per_class_recall"], f, ensure_ascii=False, indent=2)
    save_plots(save_dir, summary)
    save_spatial_errors(save_dir, predictions, args.mode)
    print(
        f"[Test] loss={test_metrics['loss']:.4f} acc={test_metrics['acc']:.4f} "
        f"macro_f1={test_metrics['macro_f1']:.4f} weighted_f1={test_metrics['weighted_f1']:.4f}"
    )
    print(f"[Artifacts] {save_dir}")
    return summary


def main():
    args = parse_args()
    args.aux_target_class_ids = parse_class_ids(args.aux_target_classes)
    configure_warnings(args.quiet_warnings)
    set_seed(args.seed)
    save_dir = PROJECT_ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    train_ds, train_loader, val_loader, test_loader, mean, std = build_loaders(args)
    (save_dir / "gnss_normalization.json").write_text(
        json.dumps({"columns": train_ds.feature_names, "mean": mean.tolist(), "std": std.tolist()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    device = torch.device(args.device)
    model = AblationModel(
        args.mode,
        train_ds.feature_dim,
        args.hidden_size,
        args.rnn_layers,
        args.dropout,
        args.pretrained,
        args.fusion,
        args.fusion_layers,
        args.fusion_heads,
        args.num_latents,
        args.image_temporal_pool,
        args.image_temporal_delta,
        args.image_window_size,
        args.aux_target_class_ids,
    ).to(device)
    load_unimodal_initialization(model, args, device)
    if args.mode == "multimodal" and args.freeze_encoders_epochs > 0:
        set_multimodal_encoder_trainable(model, False)
        print(f"[Ablation] Freeze multimodal encoders for {args.freeze_encoders_epochs} warmup epochs")
    if args.all_gpus and device.type == "cuda" and torch.cuda.device_count() > 1:
        device_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()] if args.gpu_ids else None
        model = nn.DataParallel(model, device_ids=device_ids)
        print(f"[Ablation] DataParallel GPUs: {device_ids if device_ids is not None else list(range(torch.cuda.device_count()))}")
    else:
        print(f"[Ablation] device: {device}")

    criterion = nn.CrossEntropyLoss(weight=class_weights(train_ds, device, args.class_weight_power))
    aux_criterion = None
    if args.aux_loss_weight > 0 and args.aux_target_class_ids:
        aux_criterion = nn.BCEWithLogitsLoss(
            pos_weight=aux_pos_weights(train_ds, args.aux_target_class_ids, device, args.aux_pos_weight_power)
        )
    if args.eval_checkpoint:
        checkpoint_path = Path(args.eval_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        target_model = model.module if isinstance(model, nn.DataParallel) else model
        target_model.load_state_dict(best_ckpt["model_state"])
        best_val = float(best_ckpt.get("best_val_macro_f1", -1.0))
        print(f"[Ablation] Evaluating checkpoint from epoch {best_ckpt.get('epoch', 'unknown')} best_val_macro_f1={best_val:.4f}")
        write_evaluation_artifacts(save_dir, args, model, test_loader, criterion, aux_criterion, device, best_val, [])
        return

    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=args.lr, weight_decay=args.weight_decay)
    best_val = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        if args.mode == "multimodal" and args.freeze_encoders_epochs > 0 and epoch == args.freeze_encoders_epochs + 1:
            set_multimodal_encoder_trainable(model, True)
            optimizer = torch.optim.AdamW(trainable_parameters(model), lr=args.lr, weight_decay=args.weight_decay)
            print("[Ablation] Unfroze multimodal encoders")
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, aux_criterion, device, True, args)
        val_metrics = run_epoch(model, val_loader, None, criterion, aux_criterion, device, False, args)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"[Epoch {epoch:03d}] train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} train_macro_f1={train_metrics['macro_f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            state = unwrap_model(model).state_dict()
            torch.save({"model_state": state, "args": vars(args), "epoch": epoch, "best_val_macro_f1": best_val}, save_dir / "best.pt")
        if args.early_stop_val_macro_f1 > 0 and best_val >= args.early_stop_val_macro_f1:
            print(
                f"[Ablation] Early stop: best_val_macro_f1={best_val:.4f} "
                f">= target {args.early_stop_val_macro_f1:.4f}"
            )
            break

    best_ckpt = torch.load(save_dir / "best.pt", map_location=device, weights_only=False)
    target_model = model.module if isinstance(model, nn.DataParallel) else model
    target_model.load_state_dict(best_ckpt["model_state"])
    print(f"[Ablation] Loaded best checkpoint from epoch {best_ckpt['epoch']} for test/visualization")

    write_evaluation_artifacts(save_dir, args, model, test_loader, criterion, aux_criterion, device, best_val, history)


if __name__ == "__main__":
    main()
