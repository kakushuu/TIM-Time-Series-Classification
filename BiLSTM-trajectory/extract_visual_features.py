#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract visual features from MBT model for BiLSTM training
从 MBT 模型提取视觉特征供 BiLSTM 训练使用
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

BILSTM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BILSTM_DIR.parent
MBT_DIR = PROJECT_ROOT / 'Multimodal-Fusion-with-Attention-Bottlenecks-main' / 'MBT'

# Add MBT model path
sys.path.insert(0, str(MBT_DIR))

def extract_visual_features(
    csv_file=str(PROJECT_ROOT / 'data' / 'aligned_output' / 'aligned_data.csv'),
    data_dir=str(PROJECT_ROOT),
    output_file=str(BILSTM_DIR / 'data' / 'visual_features.npz'),
    batch_size=32
):
    """
    Extract visual features from video frames using ViT-B16

    Args:
        csv_file: Path to aligned_data.csv
        data_dir: Root data directory
        output_file: Output file for extracted features
        batch_size: Batch size for feature extraction
    """
    from timm import create_model

    # Load ViT-B16 backbone (same as MBT model)
    print("Loading ViT-B16 backbone (same as MBT model)...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vit = create_model('vit_base_patch16_224_in21k', pretrained=True)
    vit.pre_logits = nn.Identity()
    vit.head = nn.Identity()
    vit = vit.to(device)
    vit.eval()

    # Feature storage
    all_features = []
    all_filepaths = []
    csv_path = Path(csv_file)
    root_dir = Path(data_dir)
    df = pd.read_csv(csv_path)

    print("Extracting visual features...")
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            frame_path = None

            if 'frame_path' in df.columns and pd.notna(row['frame_path']):
                candidate = Path(row['frame_path'])
                frame_path = candidate if candidate.is_absolute() else root_dir / candidate
            elif 'video_file' in df.columns:
                video_file = row['video_file']
                frame_number = row.get('frame_number', row.get('帧序号'))
                if pd.isna(video_file) or pd.isna(frame_number):
                    print(f"⚠ Missing frame reference at row {idx}")
                else:
                    clip_name = Path(str(video_file)).stem
                    frame_path = root_dir / 'rgb_frames' / clip_name / f"frame_{int(frame_number):04d}.jpg"

            if frame_path is None or not frame_path.exists():
                missing_path = str(frame_path) if frame_path is not None else f"row {idx}"
                print(f"⚠ Frame not found: {missing_path}")
                # Use zero features for missing frames
                all_features.append(np.zeros(768, dtype=np.float32))
                all_filepaths.append(missing_path)
                continue

            try:
                # Load and preprocess image
                img = Image.open(frame_path).convert('RGB')
                # Resize to 224x224 (ViT input size)
                img = img.resize((224, 224), Image.BILINEAR)
                img_array = np.array(img).astype(np.float32) / 255.0
                # Normalize with ImageNet stats
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_array = (img_array - mean) / std
                # Convert to tensor [1, 3, 224, 224]
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)

                # Extract features
                features = vit(img_tensor)  # [1, 768]

                all_features.append(features.cpu().numpy().squeeze())
                all_filepaths.append(str(frame_path))

            except Exception as e:
                print(f"⚠ Error processing {frame_path}: {e}")
                all_features.append(np.zeros(768, dtype=np.float32))
                all_filepaths.append(str(frame_path))

    # Convert to numpy array
    all_features = np.array(all_features, dtype=np.float32)
    print(f"\nExtracted features shape: {all_features.shape}")

    # Save features
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez_compressed(
        output_file,
        features=all_features,
        filepaths=np.array(all_filepaths)
    )
    print(f"✓ Saved to {output_file}")

    return all_features


def integrate_features_to_loader(
    features_file=str(BILSTM_DIR / 'data' / 'visual_features.npz'),
    loader_file=str(BILSTM_DIR / 'utils' / 'loader.py')
):
    """
    Modify loader.py to use real visual features instead of random ones

    This function creates a modified version of get_data_multimodal that loads
    real features from the .npz file
    """
    print(f"\n{'='*70}")
    print("Integration Instructions")
    print(f"{'='*70}")
    print(f"\n1. Features extracted to: {features_file}")
    print(f"2. Modify {loader_file}:")
    print("\n   Replace this code (lines ~151-161):")
    print("   ```python")
    print("   # Placeholder: random image features")
    print("   def generate_img_features(n_samples):")
    print("       return torch.randn(n_samples, img_feat_size).to(device)")
    print("   X_img_train = generate_img_features(X_train.shape[0])")
    print("   ```")
    print("\n   With this code:")
    print("   ```python")
    print("   # Load real visual features")
    print("   features_data = np.load('/home/research/Agri-MBT/BiLSTM-trajectory/data/visual_features.npz')")
    print("   all_img_features = torch.from_numpy(features_data['features']).float().to(device)")
    print("   ")
    print("   # Split features according to trajectory split")
    print("   # Need to track indices from get_data_trajectory")
    print("   X_img_train = all_img_features[train_indices]")
    print("   X_img_valid = all_img_features[valid_indices]")
    print("   X_img_test = all_img_features[test_indices]")
    print("   ```")
    print("\n3. Re-run training:")
    print("   python train.py --mode multimodal --epochs 50 --batch-size 64")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Extract visual features from MBT model')
    parser.add_argument('--csv', default=str(PROJECT_ROOT / 'data' / 'aligned_output' / 'aligned_data.csv'),
                        help='Path to aligned_data.csv')
    parser.add_argument('--data-dir', default=str(PROJECT_ROOT),
                        help='Root data directory')
    parser.add_argument('--output', default=str(BILSTM_DIR / 'data' / 'visual_features.npz'),
                        help='Output features file')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for feature extraction')

    args = parser.parse_args()

    # Extract features
    features = extract_visual_features(
        csv_file=args.csv,
        data_dir=args.data_dir,
        output_file=args.output,
        batch_size=args.batch_size
    )

    # Show integration instructions
    integrate_features_to_loader(args.output)
