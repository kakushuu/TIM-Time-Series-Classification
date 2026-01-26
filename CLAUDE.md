# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Agri-MBT** implements Multimodal Bottleneck Transformer for agricultural multimodal fusion (video + trajectory data). Based on the "Multimodal Fusion with Attention Bottlenecks" paper, this project adapts the audio-visual classification architecture for agricultural activity recognition using video frames and trajectory features.

## Core Architecture

### Main Components

**Multimodal Model** (`MBT/models/visual_model.py`):
- `AVmodel` class: Main multimodal architecture combining visual and trajectory streams
- Uses ViT-B16 as backbone for visual processing
- Implements both full fine-tuning and AdaptFormer (parameter-efficient transfer) modes
- 12 transformer blocks with 4 latent tokens for cross-modal attention

**Parameter-Efficient Transfer** (`MBT/models/pet_modules.py`):
- `VanillaEncoder`: Standard transformer encoder
- `AdaptFormer`: Low-rank adapter modules (reduced training cost)

**Data Loading** (`MBT/dataloader/av_data.py`):
- `AV_Dataset` loads RGB frame sequences and trajectory features
- Video: 8 frames per clip, 224x224 resolution
- Trajectory: 36 engineered features converted to 6x6 feature maps
- Requires pre-calculated normalization statistics

## Development Commands

### Training
```bash
# Run training in MBT directory
cd /home/research/Agri-MBT/Multimodal-Fusion-with-Attention-Bottlenecks-main/MBT/
python train_test.py

# Key parameters:
# --lr 3e-4           Learning rate
# --batch_size 8      Batch size
# --num_epochs 15     Training epochs
# --num_classes 11    Agricultural activity classes
# --adapter_dim 8     Adapter dimension for AdaptFormer
# --num_latent 4      Bottleneck tokens
```

### Data Requirements

**Video Structure**:
```
rgb_frames/
├── clip1/
│   ├── frame1.jpg
│   ├── frame2.jpg
│   └── ...
└── clip2/
    ├── ...
```

**Trajectory Data**:
- Excel file in `data/trajectory/` with 36 features per sample
- Processed into 6x6 feature maps for fusion with visual stream

**Annotations**:
- CSV format: `[file_name, label]`
- 11 agricultural activity classes

## Key Technical Details

### Model Adaptation
- Original: Audio spectrograms + RGB frames → Classification
- Current: Trajectory features + Video frames → Agricultural activity classification
- Trajectory branch: 36 features → 6x6 feature maps using conv2d projection

### Training Strategy
- Freeze backbone layers by default
- Train classification head and positional embeddings
- Use AdaptFormer mode for efficiency (lower memory)
- Batch size 8 recommended for GPU memory constraints

### Architecture Modifications
From `project.md`, the key adaptation:
1. Replace audio processing with trajectory feature extraction
2. Keep ViT backbone for visual stream
3. Implement trajectory branch as 6x6 feature maps
4. Use MBT fusion for multimodal interaction

## Dependencies

```bash
pip install torch torchvision torchaudio pandas numpy pillow timm
```

## Important Notes

- The project is in setup phase - data directories exist but are mostly empty
- Current implementation uses placeholder audio-related code that needs replacement with trajectory processing
- Pre-calculated normalization statistics required for both video frames and trajectory features
- GPU required for training (CUDA support)