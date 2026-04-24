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
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-agri-mbt")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from transformers import ASTFeatureExtractor, ASTModel

from dataset import CLASS_NAMES, ENGINEERED_FEATURE_NAMES, GNSS_COLS, IMAGENET_MEAN, IMAGENET_STD, PROJECT_ROOT
from models.visual_encoder import VisualEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Ablation training")
    parser.add_argument("--mode", choices=["trajectory_only", "image_only", "audio_only", "multimodal", "trimodal"], required=True)
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
    parser.add_argument("--image-sampling", choices=["center", "uniform", "nearest_causal"], default="center")
    parser.add_argument("--image-radius", type=int, default=8, help="Center sampling radius in seconds")
    parser.add_argument("--image-radius-mode", choices=["fixed", "duration"], default="fixed")
    parser.add_argument("--image-radius-duration-scale", type=float, default=0.5)
    parser.add_argument("--image-radius-classes", default="", help="Comma-separated class ids that use duration-based image radius; empty means all classes")
    parser.add_argument("--image-temporal-pool", choices=["mean", "transformer", "gru"], default="mean")
    parser.add_argument("--image-temporal-delta", choices=["none", "diff"], default="none")
    parser.add_argument("--image-augmentation", choices=["none", "light", "strong"], default="none")
    parser.add_argument("--image-frame-dropout", type=float, default=0.0, help="Per-frame replacement probability for train-time image clips")
    parser.add_argument("--image-jpeg-draft-size", type=int, default=0, help="If >0, request lower-resolution JPEG decoding before image transforms")
    parser.add_argument("--freeze-image-visual", action=argparse.BooleanOptionalAction, default=False, help="Freeze the full visual encoder, including adapters")
    parser.add_argument("--image-visual-lr", type=float, default=0.0, help="Optional lower LR for image_encoder.visual trainable parameters")
    parser.add_argument("--ast-model-name", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--freeze-audio-encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--feature-mode", choices=["raw", "engineered"], default="engineered")
    parser.add_argument("--traj-encoder", choices=["lstm", "atrnet", "trnet_seq"], default="lstm")
    parser.add_argument("--traj-feature-map-size", type=int, default=6)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight-power", type=float, default=0.5, help="Use inverse class frequency raised to this power")
    parser.add_argument("--loss-type", choices=["weighted_ce", "focal", "cb_focal"], default="weighted_ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--cb-beta", type=float, default=0.999)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--train-sampler", choices=["shuffle", "class_balanced", "class_boost"], default="shuffle")
    parser.add_argument("--sampler-weight-power", type=float, default=0.5)
    parser.add_argument("--sampler-boost-classes", default="", help="Comma-separated class ids to upsample when --train-sampler class_boost")
    parser.add_argument("--sampler-boost-factor", type=float, default=1.0)
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
    parser.add_argument("--fusion", choices=["concat", "mbt", "class_gate"], default="concat")
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--fusion-heads", type=int, default=8)
    parser.add_argument("--num-latents", type=int, default=4)
    parser.add_argument("--freeze-encoders-epochs", type=int, default=0)
    parser.add_argument("--init-traj-checkpoint", default="")
    parser.add_argument("--init-image-checkpoint", default="")
    parser.add_argument("--init-audio-checkpoint", default="")
    parser.add_argument("--eval-checkpoint", default="", help="Evaluate an existing checkpoint and write summary artifacts without training")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visual-pretrained-path", default="", help="Local ViT-B/16 checkpoint path; avoids Hugging Face downloads")
    parser.add_argument("--max-time-gap", type=float, default=1.0, help="Maximum adjacent timestamp gap, in seconds, allowed inside a trajectory window")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--batch-timing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--early-stop-val-macro-f1", type=float, default=0.0)
    parser.add_argument("--early-stop-patience", type=int, default=0, help="Stop after N epochs without validation macro-F1 improvement")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--overfit-stop-patience", type=int, default=0, help="Stop after N consecutive overfitting signals")
    parser.add_argument("--overfit-stop-min-epoch", type=int, default=0)
    parser.add_argument("--overfit-stop-gap", type=float, default=0.0, help="Train macro-F1 minus val macro-F1 threshold")
    parser.add_argument("--overfit-stop-val-loss-rise", type=float, default=0.0, help="Val loss rise over the best seen val loss")
    parser.add_argument("--temporal-smoothing", choices=["none", "min_duration"], default="none")
    parser.add_argument("--smooth-classes", default="", help="Comma-separated class ids affected by min-duration smoothing")
    parser.add_argument("--smooth-min-duration", type=int, default=5, help="Minimum prediction run duration in eval samples/seconds")
    parser.add_argument("--skip-part-diagnostics", action=argparse.BooleanOptionalAction, default=False)
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
        image_frame_dropout: float,
        image_jpeg_draft_size: int,
        feature_mode: str,
        mode: str,
        is_train: bool,
        max_time_gap: float,
        audio_sample_rate: int,
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
        self.image_frame_dropout = max(0.0, min(float(image_frame_dropout), 0.95))
        self.image_jpeg_draft_size = max(0, int(image_jpeg_draft_size))
        self.feature_mode = feature_mode
        self.mode = mode
        self.use_audio = mode in {"audio_only", "trimodal"}
        if self.use_audio and "audio_path" not in self.df.columns:
            raise ValueError(f"audio mode requires audio_path column in {path}")
        self.audio_sample_rate = int(audio_sample_rate)
        self.max_time_gap = float(max_time_gap)
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
            f"image_radius={image_radius_mode}:{image_radius} max_time_gap={self.max_time_gap:g}s"
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

    def _iter_contiguous_segments(self, seconds: np.ndarray, frame_times: np.ndarray | None = None):
        start = 0
        max_gap = float(self.max_time_gap)
        for pos in range(1, len(seconds)):
            second_gap = float(seconds[pos] - seconds[pos - 1])
            time_gap = second_gap
            if frame_times is not None:
                delta = frame_times[pos] - frame_times[pos - 1]
                time_gap = float(delta / np.timedelta64(1, "s"))
            if second_gap > max_gap or time_gap > max_gap:
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
            frame_times = group["frame_time"].to_numpy(dtype="datetime64[ns]")
            if len(idxs) == 0:
                continue
            for seg_start, seg_end in self._iter_contiguous_segments(seconds, frame_times):
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
            idxs = group.index.to_numpy()
            seconds = group["second_in_video"].to_numpy()
            frame_times = group["frame_time"].to_numpy(dtype="datetime64[ns]")
            for seg_start, seg_end in self._iter_contiguous_segments(seconds, frame_times):
                seg = group.iloc[seg_start:seg_end + 1]
                raw = seg[GNSS_COLS].values.astype(np.float32)
                features[idxs[seg_start:seg_end + 1]] = self._make_features(raw)
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
        elif self.image_sampling == "nearest_causal":
            candidates = list(range(0, center_pos + 1))
            positions = candidates[-self.image_window_size:]
            if len(positions) < self.image_window_size:
                positions = [positions[0] if positions else center_pos] * (self.image_window_size - len(positions)) + positions
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
                img = Image.open(img_path)
                if self.image_jpeg_draft_size > 0:
                    img.draft("RGB", (self.image_jpeg_draft_size, self.image_jpeg_draft_size))
                img = img.convert("RGB")
            else:
                img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            frames.append(self.transform(img))
        clip = torch.stack(frames, dim=0)
        if self.is_train and self.image_frame_dropout > 0 and clip.size(0) > 1:
            drop_mask = torch.rand(clip.size(0)) < self.image_frame_dropout
            if bool(drop_mask.all()):
                drop_mask[int(torch.randint(0, clip.size(0), (1,)).item())] = False
            if bool(drop_mask.any()):
                ref_idx = clip.size(0) - 1 if self.image_sampling == "nearest_causal" else clip.size(0) // 2
                clip[drop_mask] = clip[ref_idx].clone()
        return clip

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
        if self.mode in {"trajectory_only", "audio_only"}:
            images = torch.empty(0)
        else:
            images = self._load_images(indices, anchor_slot)
        if self.use_audio:
            row = self.df.iloc[center_idx]
            audio_path = Path(str(row["audio_path"]))
            if not audio_path.is_absolute():
                audio_path = self.project_root / audio_path
            waveform, sample_rate = torchaudio.load(str(audio_path))
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if int(sample_rate) != self.audio_sample_rate:
                waveform = torchaudio.functional.resample(waveform, int(sample_rate), self.audio_sample_rate)
            audio = waveform.squeeze(0)
            target_len = self.audio_sample_rate
            if audio.numel() < target_len:
                audio = torch.nn.functional.pad(audio, (0, target_len - audio.numel()))
            elif audio.numel() > target_len:
                audio = audio[:target_len]
        else:
            audio = torch.empty(0)
        label = int(self.df.iloc[center_idx]["分类"])
        return traj, images, audio, label


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.score(torch.tanh(x)).squeeze(-1), dim=1)
        return (x * weights.unsqueeze(-1)).sum(dim=1)


class TrajectoryATRNetEncoder(nn.Module):
    """ATRNet-style pointwise trajectory encoder using a 6x6 feature map."""

    def __init__(self, input_dim: int, feature_map_size: int = 6, dropout: float = 0.35):
        super().__init__()
        if feature_map_size * feature_map_size != input_dim:
            raise ValueError(
                f"ATRNet trajectory encoder expects a square feature map, got input_dim={input_dim} "
                f"and feature_map_size={feature_map_size}"
            )
        self.feature_map_size = int(feature_map_size)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        pooled_size = self.feature_map_size // 2
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * pooled_size * pooled_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(128),
        )

    def forward(self, x):
        point_features = x[:, -1, :]
        feature_map = point_features.reshape(point_features.size(0), 1, self.feature_map_size, self.feature_map_size)
        return self.head(self.conv(feature_map))


class TrajectoryTRNetSequenceEncoder(nn.Module):
    """TRNet-style feature-map CNN per trajectory point followed by temporal BiLSTM attention."""

    def __init__(self, input_dim: int, hidden_size: int = 256, layers: int = 2, feature_map_size: int = 6, dropout: float = 0.35):
        super().__init__()
        if feature_map_size * feature_map_size != input_dim:
            raise ValueError(
                f"TRNet sequence encoder expects a square feature map, got input_dim={input_dim} "
                f"and feature_map_size={feature_map_size}"
            )
        self.feature_map_size = int(feature_map_size)
        self.hidden_size = int(hidden_size)
        self.point_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        pooled_size = self.feature_map_size // 2
        self.point_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * pooled_size * pooled_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )
        self.rnn = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.out = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(dropout))
        self.pool = AttentionPool(hidden_size)

    def forward_tokens(self, x):
        batch_size, seq_len, feature_dim = x.shape
        point_maps = x.reshape(batch_size * seq_len, 1, self.feature_map_size, self.feature_map_size)
        point_feats = self.point_proj(self.point_cnn(point_maps)).reshape(batch_size, seq_len, self.hidden_size)
        out, _ = self.rnn(point_feats)
        out = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]
        return self.out(out)

    def forward(self, x):
        return self.pool(self.forward_tokens(x))


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
    def __init__(
        self,
        pretrained=False,
        pretrained_path="",
        dropout=0.35,
        temporal_pool="mean",
        temporal_delta="none",
        max_frames=32,
        freeze_visual: bool = False,
    ):
        super().__init__()
        self.visual = VisualEncoder(pretrained=pretrained, pretrained_path=pretrained_path)
        if freeze_visual:
            for param in self.visual.parameters():
                param.requires_grad = False
            print("[ImageEncoder] Frozen full visual encoder, including adapters")
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


class AudioEncoder(nn.Module):
    def __init__(self, model_name: str, sample_rate: int = 16000, dropout: float = 0.35, freeze: bool = False):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
        self.ast = ASTModel.from_pretrained(model_name)
        hidden_size = int(self.ast.config.hidden_size)
        if freeze:
            for param in self.ast.parameters():
                param.requires_grad = False
        self.out = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(dropout))

    def _preprocess(self, audio: torch.Tensor) -> torch.Tensor:
        audio_np = [item.detach().cpu().numpy() for item in audio]
        features = self.feature_extractor(
            audio_np,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        return features["input_values"].to(audio.device)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        input_values = self._preprocess(audio)
        outputs = self.ast(input_values=input_values)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        return self.out(pooled)


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


class ClassAdaptiveLogitFusion(nn.Module):
    """Per-class modality gating over modality-specific logits.

    The concat baseline lets a single classifier consume all modality features at
    once. This module keeps a separate class-logit head for each modality and
    learns a softmax gate for every class, so audio-dominant process states can
    keep audio evidence while motion/scene classes can rely more on trajectory
    or image features.
    """

    def __init__(self, modality_dims, num_classes=11, hidden_dim=512, gate_dim=256, dropout=0.35):
        super().__init__()
        self.num_modalities = len(modality_dims)
        self.num_classes = int(num_classes)
        if self.num_modalities < 2:
            raise ValueError("ClassAdaptiveLogitFusion requires at least two modalities")

        self.modality_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Dropout(dropout),
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_classes),
            )
            for dim in modality_dims
        ])
        self.gate_projs = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, gate_dim),
                nn.GELU(),
            )
            for dim in modality_dims
        ])
        self.gate_hidden = nn.Sequential(
            nn.LayerNorm(gate_dim * self.num_modalities),
            nn.Linear(gate_dim * self.num_modalities, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate_out = nn.Linear(hidden_dim, self.num_classes * self.num_modalities)
        nn.init.zeros_(self.gate_out.weight)
        nn.init.zeros_(self.gate_out.bias)

    def forward(self, features):
        if len(features) != self.num_modalities:
            raise ValueError(f"expected {self.num_modalities} modality features, got {len(features)}")
        logits = torch.stack(
            [head(feat) for head, feat in zip(self.modality_heads, features)],
            dim=-1,
        )
        gate_input = torch.cat(
            [proj(feat) for proj, feat in zip(self.gate_projs, features)],
            dim=-1,
        )
        gates = self.gate_out(self.gate_hidden(gate_input))
        gates = gates.view(-1, self.num_classes, self.num_modalities)
        gates = torch.softmax(gates, dim=-1)
        return (logits * gates).sum(dim=-1)


class AblationModel(nn.Module):
    def __init__(self, mode, traj_dim, hidden_size, layers, dropout, pretrained, visual_pretrained_path, fusion, fusion_layers, fusion_heads, num_latents, image_temporal_pool, image_temporal_delta, image_window_size, aux_target_classes, ast_model_name, audio_sample_rate, freeze_audio_encoder, freeze_image_visual, traj_encoder_type, traj_feature_map_size):
        super().__init__()
        self.mode = mode
        self.fusion_mode = fusion
        self.traj_encoder_type = traj_encoder_type
        self.aux_target_classes = list(aux_target_classes)
        if mode in {"trajectory_only", "multimodal", "trimodal"}:
            if traj_encoder_type == "atrnet":
                self.traj_encoder = TrajectoryATRNetEncoder(
                    traj_dim,
                    feature_map_size=traj_feature_map_size,
                    dropout=dropout,
                )
                traj_out_dim = 128
            elif traj_encoder_type == "trnet_seq":
                self.traj_encoder = TrajectoryTRNetSequenceEncoder(
                    traj_dim,
                    hidden_size=hidden_size,
                    layers=layers,
                    feature_map_size=traj_feature_map_size,
                    dropout=dropout,
                )
                traj_out_dim = hidden_size
            else:
                self.traj_encoder = TrajectoryEncoder(traj_dim, hidden_size, layers, dropout)
                traj_out_dim = hidden_size
        else:
            traj_out_dim = 0
        if mode in {"image_only", "multimodal", "trimodal"}:
            self.image_encoder = ImageEncoder(
                pretrained=pretrained,
                pretrained_path=visual_pretrained_path,
                dropout=dropout,
                temporal_pool=image_temporal_pool,
                temporal_delta=image_temporal_delta,
                max_frames=max(image_window_size, 32),
                freeze_visual=freeze_image_visual,
            )
        if mode in {"audio_only", "trimodal"}:
            self.audio_encoder = AudioEncoder(
                ast_model_name,
                sample_rate=audio_sample_rate,
                dropout=dropout,
                freeze=freeze_audio_encoder,
            )
        if mode == "trajectory_only":
            in_dim = traj_out_dim
        elif mode == "image_only":
            in_dim = 768
        elif mode == "audio_only":
            in_dim = 768
        elif fusion == "class_gate":
            if mode not in {"multimodal", "trimodal"}:
                raise ValueError("--fusion class_gate requires --mode multimodal or --mode trimodal")
            modality_dims = []
            if mode in {"multimodal", "trimodal"}:
                modality_dims.append(traj_out_dim)
                modality_dims.append(768)
            if mode == "trimodal":
                modality_dims.append(768)
            self.class_gate_fusion = ClassAdaptiveLogitFusion(
                modality_dims,
                num_classes=11,
                hidden_dim=512,
                gate_dim=256,
                dropout=dropout,
            )
            in_dim = 0
        elif fusion == "mbt":
            if traj_encoder_type != "lstm":
                raise ValueError("--fusion mbt currently requires --traj-encoder lstm")
            if mode == "trimodal":
                raise ValueError("--fusion mbt is only implemented for trajectory+video multimodal; use --fusion concat for trimodal")
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
            in_dim = traj_out_dim + 768 + (768 if mode == "trimodal" else 0)
        if fusion == "class_gate":
            self.classifier = None
            self.aux_classifier = None
        else:
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

    def forward(self, traj, images, audio):
        if self.mode == "multimodal" and self.fusion_mode == "mbt":
            traj_tokens = self.traj_encoder.forward_tokens(traj)
            image_tokens = self.image_encoder.forward_tokens(images)
            return self._emit_logits(self.mbt_fusion(traj_tokens, image_tokens))
        feats = []
        if self.mode in {"trajectory_only", "multimodal", "trimodal"}:
            feats.append(self.traj_encoder(traj))
        if self.mode in {"image_only", "multimodal", "trimodal"}:
            feats.append(self.image_encoder(images))
        if self.mode in {"audio_only", "trimodal"}:
            feats.append(self.audio_encoder(audio))
        if self.fusion_mode == "class_gate" and self.mode in {"multimodal", "trimodal"}:
            return self.class_gate_fusion(feats)
        return self._emit_logits(torch.cat(feats, dim=1) if len(feats) > 1 else feats[0])


def build_image_transform(args, train: bool):
    img_size = args.img_size if hasattr(args, "img_size") else 224
    if not train or args.image_augmentation == "none":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    if args.image_augmentation == "light":
        crop_scale = (0.78, 1.0)
        jitter = transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.02)
        blur_p = 0.08
        erase_p = 0.12
    else:
        crop_scale = (0.62, 1.0)
        jitter = transforms.ColorJitter(brightness=0.30, contrast=0.30, saturation=0.22, hue=0.035)
        blur_p = 0.18
        erase_p = 0.25

    resize_side = max(img_size + 32, 256)
    return transforms.Compose([
        transforms.Resize(resize_side),
        transforms.RandomResizedCrop((img_size, img_size), scale=crop_scale, ratio=(0.9, 1.1)),
        transforms.RandomApply([jitter], p=0.8),
        transforms.RandomGrayscale(p=0.05 if args.image_augmentation == "light" else 0.10),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=blur_p),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=erase_p, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
    ])


def build_loaders(args):
    train_transform = build_image_transform(args, True)
    eval_transform = build_image_transform(args, False)
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
        args.image_frame_dropout,
        args.image_jpeg_draft_size,
        args.feature_mode,
        args.mode,
        True,
        args.max_time_gap,
        args.audio_sample_rate,
        train_transform,
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
        0.0,
        args.image_jpeg_draft_size,
        args.feature_mode,
        args.mode,
        False,
        args.max_time_gap,
        args.audio_sample_rate,
        eval_transform,
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
        0.0,
        args.image_jpeg_draft_size,
        args.feature_mode,
        args.mode,
        False,
        args.max_time_gap,
        args.audio_sample_rate,
        eval_transform,
        mean,
        std,
    )
    kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    train_sampler = None
    train_shuffle = True
    if args.train_sampler in {"class_balanced", "class_boost"}:
        labels = torch.tensor([int(train_ds.df.iloc[center]["分类"]) for _, center, _ in train_ds.windows], dtype=torch.long)
        counts = torch.bincount(labels, minlength=11).float().clamp(min=1.0)
        if args.train_sampler == "class_balanced":
            sample_weights = counts[labels].pow(-float(args.sampler_weight_power))
        else:
            sample_weights = counts[labels].pow(-float(args.sampler_weight_power)) if args.sampler_weight_power > 0 else torch.ones_like(labels, dtype=torch.float32)
            boost_classes = set(parse_class_ids(args.sampler_boost_classes))
            if boost_classes and args.sampler_boost_factor > 1:
                mask = torch.tensor([int(label.item()) in boost_classes for label in labels], dtype=torch.bool)
                sample_weights[mask] *= float(args.sampler_boost_factor)
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


def effective_num_class_weights(dataset, device, beta: float):
    labels = [int(dataset.df.iloc[center]["分类"]) for _, center, _ in dataset.windows]
    counts = torch.bincount(torch.tensor(labels), minlength=11).float().clamp(min=1.0)
    beta = float(beta)
    if beta <= 0:
        weights = counts.reciprocal()
    else:
        weights = (1.0 - beta) / (1.0 - torch.pow(torch.full_like(counts, beta), counts))
    return (weights / weights.sum() * 11).to(device)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("weight", weight.detach().clone() if weight is not None else None)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, labels):
        log_probs = torch.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        num_classes = logits.size(1)
        targets = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1.0)
        if self.label_smoothing > 0:
            smooth = min(max(self.label_smoothing, 0.0), 1.0)
            targets = targets * (1.0 - smooth) + smooth / num_classes
        focal = torch.pow(1.0 - probs, self.gamma)
        loss = -targets * focal * log_probs
        if self.weight is not None:
            loss = loss * self.weight.view(1, -1)
        return loss.sum(dim=1).mean()


def build_criterion(args, train_ds, device):
    if args.loss_type == "weighted_ce":
        return nn.CrossEntropyLoss(
            weight=class_weights(train_ds, device, args.class_weight_power),
            label_smoothing=float(args.label_smoothing),
        )
    if args.loss_type == "focal":
        return FocalLoss(
            weight=class_weights(train_ds, device, args.class_weight_power),
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
        )
    if args.loss_type == "cb_focal":
        return FocalLoss(
            weight=effective_num_class_weights(train_ds, device, args.cb_beta),
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
        )
    raise ValueError(f"unknown loss type: {args.loss_type}")


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


def build_optimizer(model, args):
    named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if (
        args.mode in {"image_only", "multimodal", "trimodal"}
        and float(args.image_visual_lr) > 0
    ):
        visual_params = []
        other_params = []
        for name, param in named_params:
            clean_name = name[len("module."):] if name.startswith("module.") else name
            if clean_name.startswith("image_encoder.visual."):
                visual_params.append(param)
            else:
                other_params.append(param)
        groups = []
        if visual_params:
            groups.append({"params": visual_params, "lr": float(args.image_visual_lr)})
        if other_params:
            groups.append({"params": other_params, "lr": float(args.lr)})
        print(
            "[Ablation] AdamW parameter groups: "
            f"visual={sum(p.numel() for p in visual_params)/1e6:.2f}M lr={float(args.image_visual_lr):g}, "
            f"other={sum(p.numel() for p in other_params)/1e6:.2f}M lr={float(args.lr):g}"
        )
        return torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    return torch.optim.AdamW([param for _, param in named_params], lr=args.lr, weight_decay=args.weight_decay)


def set_multimodal_encoder_trainable(model, trainable: bool):
    base = unwrap_model(model)
    for name in ["traj_encoder", "image_encoder", "audio_encoder"]:
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
    if args.mode not in {"multimodal", "trimodal"}:
        return
    load_branch_checkpoint(model, args.init_traj_checkpoint, "traj_encoder", device)
    load_branch_checkpoint(model, args.init_image_checkpoint, "image_encoder", device)
    load_branch_checkpoint(model, args.init_audio_checkpoint, "audio_encoder", device)


def unpack_model_output(output):
    if isinstance(output, tuple):
        return output
    return output, None


def _prediction_runs(seconds: np.ndarray, preds: np.ndarray, gap: int = 1):
    if len(preds) == 0:
        return []
    runs = []
    start = 0
    for idx in range(1, len(preds)):
        if preds[idx] != preds[idx - 1] or int(seconds[idx]) - int(seconds[idx - 1]) > gap:
            runs.append((start, idx - 1, int(preds[idx - 1])))
            start = idx
    runs.append((start, len(preds) - 1, int(preds[-1])))
    return runs


def apply_temporal_smoothing(preds: np.ndarray, dataset: LongWindowDataset, args) -> np.ndarray:
    if args.temporal_smoothing == "none" or len(preds) == 0:
        return preds
    smooth_classes = set(parse_class_ids(args.smooth_classes))
    if not smooth_classes:
        return preds
    min_duration = max(int(args.smooth_min_duration), 1)
    meta = dataset.df.iloc[[center for _, center, _ in dataset.windows]][["video_file", "second_in_video"]].reset_index(drop=True)
    smoothed = preds.copy()
    for _, group in meta.groupby("video_file", sort=False):
        order = group.index.to_numpy()
        seconds = group["second_in_video"].astype(int).to_numpy()
        local_preds = smoothed[order].copy()
        runs = _prediction_runs(seconds, local_preds)
        replacements = {}
        for run_idx, (start, end, pred) in enumerate(runs):
            if pred not in smooth_classes:
                continue
            duration = int(seconds[end] - seconds[start] + 1)
            if duration >= min_duration:
                continue
            left = runs[run_idx - 1] if run_idx > 0 else None
            right = runs[run_idx + 1] if run_idx + 1 < len(runs) else None
            replacement = None
            if left and right and left[2] == right[2]:
                replacement = left[2]
            elif left or right:
                candidates = []
                if left:
                    candidates.append((left[1] - left[0] + 1, left[2]))
                if right:
                    candidates.append((right[1] - right[0] + 1, right[2]))
                replacement = max(candidates)[1]
            if replacement is not None:
                replacements[(start, end)] = replacement
        for (start, end), replacement in replacements.items():
            local_preds[start:end + 1] = replacement
        smoothed[order] = local_preds
    return smoothed


def run_epoch(model, loader, optimizer, criterion, aux_criterion, device, train, args):
    model.train(train)
    total_loss = total_correct = total_samples = 0
    preds_all, labels_all = [], []
    max_batches = args.max_train_batches if train else args.max_eval_batches
    last_batch_end = time.perf_counter()
    for batch_idx, (traj, images, audio, labels) in enumerate(loader, start=1):
        batch_start = time.perf_counter()
        data_time = batch_start - last_batch_end
        traj = traj.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        images = images.to(device, non_blocking=True) if args.mode in {"image_only", "multimodal", "trimodal"} else images
        audio = audio.to(device, non_blocking=True) if args.mode in {"audio_only", "trimodal"} else audio
        with torch.set_grad_enabled(train):
            logits, aux_logits = unpack_model_output(model(traj, images, audio))
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
        batch_end = time.perf_counter()
        if args.batch_timing and (batch_idx <= 2 or batch_idx % 10 == 0):
            split = "train" if train else "eval"
            print(
                f"[BatchTiming] {split} batch={batch_idx} "
                f"data_s={data_time:.3f} step_s={batch_end - batch_start:.3f} "
                f"samples={labels.size(0)}",
                flush=True,
            )
        last_batch_end = batch_end
        if max_batches and batch_idx >= max_batches:
            break
    labels_np = torch.cat(labels_all).numpy() if labels_all else np.array([])
    preds_np = torch.cat(preds_all).numpy() if preds_all else np.array([])
    if not train and len(preds_np):
        preds_np = apply_temporal_smoothing(preds_np, loader.dataset, args)
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
    raw_preds_all, labels_all = [], []
    max_batches = args.max_eval_batches
    with torch.no_grad():
        for batch_idx, (traj, images, audio, labels) in enumerate(loader, start=1):
            traj = traj.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images = images.to(device, non_blocking=True) if args.mode in {"image_only", "multimodal", "trimodal"} else images
            audio = audio.to(device, non_blocking=True) if args.mode in {"audio_only", "trimodal"} else audio
            logits, _ = unpack_model_output(model(traj, images, audio))
            raw_preds_all.append(logits.argmax(dim=1).detach().cpu())
            labels_all.append(labels.detach().cpu())
            if max_batches and batch_idx >= max_batches:
                break
    raw_preds = torch.cat(raw_preds_all).numpy() if raw_preds_all else np.array([])
    labels_np = torch.cat(labels_all).numpy() if labels_all else np.array([])
    preds = raw_preds
    if len(raw_preds) and len(raw_preds) == len(loader.dataset):
        preds = apply_temporal_smoothing(raw_preds, loader.dataset, args)
    rows = []
    for i in range(len(labels_np)):
        rows.append({
            "dataset_index": i,
            "y_true": int(labels_np[i]),
            "y_pred": int(preds[i]),
            "y_pred_raw": int(raw_preds[i]),
            "correct": bool(labels_np[i] == preds[i]),
        })
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
    for metric, title, filename in [
        ("loss", "Loss", "training_loss.png"),
        ("acc", "Accuracy", "training_accuracy.png"),
        ("macro_f1", "Macro F1", "training_macro_f1.png"),
        ("weighted_f1", "Weighted F1", "training_weighted_f1.png"),
    ]:
        fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=160)
        ax.plot(epochs, [r[f"train_{metric}"] for r in rows], label="train")
        ax.plot(epochs, [r[f"val_{metric}"] for r in rows], label="val")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_dir / filename)
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

    for metric, title, filename in [
        ("precision", "Test Per-Class Precision", "per_class_precision.png"),
        ("recall", "Test Per-Class Recall", "per_class_recall_plot.png"),
        ("f1", "Test Per-Class F1", "per_class_f1.png"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 4.5), dpi=160)
        x = np.arange(11)
        ax.bar(x, per_class_df[metric], width=0.65)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_xlabel("Class ID")
        ax.set_ylabel("Score")
        ax.set_xticks(range(11))
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(save_dir / filename)
        plt.close(fig)


def _prediction_dates(predictions: Path) -> list[str]:
    try:
        frame_times = pd.read_csv(predictions, encoding="utf-8-sig", usecols=["frame_time"])
    except Exception as exc:
        print(f"[PartDiagnostics] skipped date inference: {exc}")
        return ["2024-10-27"]
    dates = pd.to_datetime(frame_times["frame_time"], errors="coerce").dt.strftime("%Y-%m-%d")
    dates = sorted(date for date in dates.dropna().unique().tolist())
    return dates or ["2024-10-27"]


def save_part_diagnostics(save_dir: Path):
    script = PROJECT_ROOT / "scripts" / "plot_b_deep_part_prediction_maps.py"
    predictions = save_dir / "predictions.csv"
    if not script.exists() or not predictions.exists():
        return
    output_root = save_dir / "part_diagnostics"
    dates = _prediction_dates(predictions)
    summaries = []
    for date in dates:
        output_dir = output_root if len(dates) == 1 else output_root / date
        cmd = [
            sys.executable,
            str(script),
            "--predictions",
            str(predictions),
            "--date",
            date,
            "--output-dir",
            str(output_dir),
        ]
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, capture_output=True)
        if result.stdout:
            print(result.stdout.rstrip())
        if result.returncode != 0:
            print(f"[PartDiagnostics] skipped date={date}: {result.stderr.strip()}")
            continue
        summary_path = output_dir / f"{date}_part_diagnostic_summary.csv"
        if summary_path.exists():
            try:
                summary = pd.read_csv(summary_path, encoding="utf-8-sig")
                if not summary.empty:
                    summary.insert(0, "date", date)
                    summary.insert(1, "diagnostic_dir", str(output_dir.relative_to(save_dir)))
                    summaries.append(summary)
            except Exception as exc:
                print(f"[PartDiagnostics] could not read {summary_path}: {exc}")
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(
            output_root / "part_diagnostic_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )


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
    if not args.skip_part_diagnostics:
        save_part_diagnostics(save_dir)
    print(
        f"[Test] loss={test_metrics['loss']:.4f} acc={test_metrics['acc']:.4f} "
        f"macro_f1={test_metrics['macro_f1']:.4f} weighted_f1={test_metrics['weighted_f1']:.4f}"
    )
    print(f"[Artifacts] {save_dir}")
    return summary


def main():
    args = parse_args()
    args.aux_target_class_ids = parse_class_ids(args.aux_target_classes)
    if args.traj_encoder in {"atrnet", "trnet_seq"} and args.feature_mode != "engineered":
        raise ValueError(f"--traj-encoder {args.traj_encoder} requires --feature-mode engineered")
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
        args.visual_pretrained_path,
        args.fusion,
        args.fusion_layers,
        args.fusion_heads,
        args.num_latents,
        args.image_temporal_pool,
        args.image_temporal_delta,
        args.image_window_size,
        args.aux_target_class_ids,
        args.ast_model_name,
        args.audio_sample_rate,
        args.freeze_audio_encoder,
        args.freeze_image_visual,
        args.traj_encoder,
        args.traj_feature_map_size,
    ).to(device)
    load_unimodal_initialization(model, args, device)
    if args.mode in {"multimodal", "trimodal"} and args.freeze_encoders_epochs > 0:
        set_multimodal_encoder_trainable(model, False)
        print(f"[Ablation] Freeze multimodal encoders for {args.freeze_encoders_epochs} warmup epochs")
    if args.all_gpus and device.type == "cuda" and torch.cuda.device_count() > 1:
        device_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()] if args.gpu_ids else None
        model = nn.DataParallel(model, device_ids=device_ids)
        print(f"[Ablation] DataParallel GPUs: {device_ids if device_ids is not None else list(range(torch.cuda.device_count()))}")
    else:
        print(f"[Ablation] device: {device}")

    criterion = build_criterion(args, train_ds, device)
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

    optimizer = build_optimizer(model, args)
    best_val = -1.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    overfit_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        if args.mode in {"multimodal", "trimodal"} and args.freeze_encoders_epochs > 0 and epoch == args.freeze_encoders_epochs + 1:
            set_multimodal_encoder_trainable(model, True)
            optimizer = build_optimizer(model, args)
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
        best_val_loss = min(best_val_loss, float(val_metrics["loss"]))
        improved = val_metrics["macro_f1"] > best_val + float(args.early_stop_min_delta)
        if improved:
            best_val = val_metrics["macro_f1"]
            epochs_without_improvement = 0
            state = unwrap_model(model).state_dict()
            torch.save({"model_state": state, "args": vars(args), "epoch": epoch, "best_val_macro_f1": best_val}, save_dir / "best.pt")
        else:
            epochs_without_improvement += 1
        if args.early_stop_val_macro_f1 > 0 and best_val >= args.early_stop_val_macro_f1:
            print(
                f"[Ablation] Early stop: best_val_macro_f1={best_val:.4f} "
                f">= target {args.early_stop_val_macro_f1:.4f}"
            )
            break
        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(
                f"[Ablation] Early stop: no val_macro_f1 improvement > "
                f"{args.early_stop_min_delta:.4g} for {args.early_stop_patience} epochs "
                f"(best_val_macro_f1={best_val:.4f})"
            )
            break
        if args.overfit_stop_patience > 0 and epoch >= args.overfit_stop_min_epoch:
            train_val_gap = float(train_metrics["macro_f1"] - val_metrics["macro_f1"])
            val_loss_rise = float(val_metrics["loss"] - best_val_loss)
            overfit_signal = (
                train_val_gap >= float(args.overfit_stop_gap)
                and val_loss_rise >= float(args.overfit_stop_val_loss_rise)
            )
            if overfit_signal:
                overfit_epochs += 1
            else:
                overfit_epochs = 0
            if overfit_epochs >= args.overfit_stop_patience:
                print(
                    "[Ablation] Overfit stop: "
                    f"train_val_macro_f1_gap={train_val_gap:.4f} >= {float(args.overfit_stop_gap):.4f}, "
                    f"val_loss_rise={val_loss_rise:.4f} >= {float(args.overfit_stop_val_loss_rise):.4f} "
                    f"for {args.overfit_stop_patience} epochs"
                )
                break

    best_ckpt = torch.load(save_dir / "best.pt", map_location=device, weights_only=False)
    target_model = model.module if isinstance(model, nn.DataParallel) else model
    target_model.load_state_dict(best_ckpt["model_state"])
    print(f"[Ablation] Loaded best checkpoint from epoch {best_ckpt['epoch']} for test/visualization")

    write_evaluation_artifacts(save_dir, args, model, test_loader, criterion, aux_criterion, device, best_val, history)


if __name__ == "__main__":
    main()
