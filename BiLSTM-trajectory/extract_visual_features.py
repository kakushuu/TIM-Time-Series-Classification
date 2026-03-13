#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract visual features from MBT model for BiLSTM training
从 MBT 模型提取视觉特征供 BiLSTM 训练使用
"""

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add MBT model path
sys.path.insert(0, '/home/research/Agri-MBT/Multimodal-Fusion-with-Attention-Bottlenecks-main/MBT')

def extract_visual_features(
    csv_file='/home/research/Agri-MBT/data/aligned_output/aligned_data.csv',
    data_dir='/home/research/Agri-MBT',
    output_file='/home/research/Agri-MBT/BiLSTM-trajectory/data/visual_features.npz',
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

    print("Extracting visual features...")
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # Construct frame path
            video_path = row['视频路径']
            frame_idx = row['帧序号']

            # Expected path: data_dir/rgb_frames/{clip_name}/frame_{idx}.jpg
            clip_name = os.path.splitext(os.path.basename(video_path))[0]
            frame_path = os.path.join(
                data_dir,
                'rgb_frames',
                clip_name,
                f'frame_{frame_idx:04d}.jpg'
            )

            if not os.path.exists(frame_path):
                print(f"⚠ Frame not found: {frame_path}")
                # Use zero features for missing frames
                all_features.append(np.zeros(768, dtype=np.float32))
                all_filepaths.append(frame_path)
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
                all_filepaths.append(frame_path)

            except Exception as e:
                print(f"⚠ Error processing {frame_path}: {e}")
                all_features.append(np.zeros(768, dtype=np.float32))
                all_filepaths.append(frame_path)

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
    features_file='/home/research/Agri-MBT/BiLSTM-trajectory/data/visual_features.npz',
    loader_file='/home/research/Agri-MBT/BiLSTM-trajectory/utils/loader.py'
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
    parser.add_argument('--csv', default='/home/research/Agri-MBT/data/aligned_output/aligned_data.csv',
                        help='Path to aligned_data.csv')
    parser.add_argument('--data-dir', default='/home/research/Agri-MBT',
                        help='Root data directory')
    parser.add_argument('--output', default='/home/research/Agri-MBT/BiLSTM-trajectory/data/visual_features.npz',
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
