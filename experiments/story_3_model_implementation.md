# Story 3: TC-AdaptFormer Model Implementation

## User Story
As a researcher, I want a complete PyTorch implementation of the TC-AdaptFormer model that fuses video and GNSS features for agricultural activity classification, so that I can train and evaluate the multimodal fusion approach.

## Context
- **Architecture**: TC-AdaptFormer (ViT-B16 backbone + AdaptFormer + GNSS Cross-Attention)
- **Input**: Video `(B, T, 3, 224, 224)` + GNSS `(B, 7)`
- **Output**: Logits `(B, 11)` for 11 agricultural activity classes
- **Design Reference**: `docs/tc_adaptformer_architecture.md` (from Story 1)
- **Existing Code**: `/home/research/Agri-MBT/Multimodal-Fusion-with-Attention-Bottlenecks-main/MBT/models/`

## Acceptance Criteria

### AC1: GNSS Encoder Module
- [ ] Class `GNSSEncoder(nn.Module)` in `src/models/gnss_encoder.py`
- [ ] Architecture: MLP with layers `7 → 128 → 768`
- [ ] Activation: GELU between layers
- [ ] Forward: `gnss (B, 7) → query_embed (B, 768)`
- [ ] Include LayerNorm after final layer
- [ ] Trainable parameters: ~0.1M

### AC2: Visual Encoder with Frozen ViT-B16
- [ ] Class `VisualEncoder(nn.Module)` in `src/models/visual_encoder.py`
- [ ] Load pretrained ViT-B16 from `timm.create_model('vit_base_patch16_224', pretrained=True)`
- [ ] Freeze all ViT parameters: `param.requires_grad = False`
- [ ] Remove classification head: `model.head = nn.Identity()`
- [ ] Forward: `video (B, T, 3, 224, 224) → tokens (B, T, 196, 768)`
- [ ] Process T frames independently (loop or reshape)

### AC3: AdaptFormer Integration
- [ ] Reuse `AdaptFormer` class from `MBT/models/pet_modules.py`
- [ ] Insert adapters into ViT blocks (all 12 blocks)
- [ ] Adapter config: `adapter_dim=64, rank=8`
- [ ] Only adapter parameters trainable: ~2M params
- [ ] Verify: `sum(p.numel() for p in model.parameters() if p.requires_grad) ≈ 2M`

### AC4: Cross-Attention Fusion Module
- [ ] Class `CrossAttentionFusion(nn.Module)` in `src/models/fusion.py`
- [ ] Use `nn.MultiheadAttention(embed_dim=768, num_heads=12)`
- [ ] Forward signature: `fusion(query_gnss, key_visual, value_visual)`
- [ ] Input: `query (B, 768)`, `key/value (B, T*196, 768)` (flatten T and patches)
- [ ] Output: `fused_features (B, 768)` after attention pooling
- [ ] Attention pooling: mean over attended tokens

### AC5: Temporal Pooling
- [ ] Implement in `VisualEncoder` or `CrossAttentionFusion`
- [ ] Method: mean pooling across T=5 frames
- [ ] Input: `(B, T, 768)` → Output: `(B, 768)`
- [ ] Alternative: learnable weighted pooling (optional enhancement)

### AC6: Complete TC-AdaptFormer Model
- [ ] Class `TCAdaptFormer(nn.Module)` in `src/models/tc_adaptformer.py`
- [ ] Compose: GNSSEncoder + VisualEncoder + CrossAttentionFusion + Classifier
- [ ] Classifier: `nn.Linear(768, 11)`
- [ ] Forward pass:
  ```python
  def forward(self, video, gnss):
      # video: (B, T, 3, 224, 224)
      # gnss: (B, 7)
      gnss_query = self.gnss_encoder(gnss)  # (B, 768)
      visual_tokens = self.visual_encoder(video)  # (B, T*196, 768)
      fused = self.fusion(gnss_query, visual_tokens, visual_tokens)  # (B, 768)
      logits = self.classifier(fused)  # (B, 11)
      return logits
  ```

### AC7: Model Initialization
- [ ] Implement `__init__` with proper weight initialization
- [ ] Xavier/Kaiming init for MLP layers
- [ ] Load ViT-B16 pretrained weights from timm
- [ ] Set all ViT params to `requires_grad=False`
- [ ] Print trainable parameter count on initialization

### AC8: Model Summary Function
- [ ] Implement `model.summary()` method
- [ ] Print architecture overview:
  - Total parameters
  - Trainable parameters
  - Frozen parameters
  - Memory footprint estimate (MB)
- [ ] Use `torchsummary` or custom implementation

### AC9: Save/Load Checkpoints
- [ ] Implement `save_checkpoint(model, optimizer, epoch, path)`
- [ ] Implement `load_checkpoint(path, model, optimizer=None)`
- [ ] Save: model state_dict, optimizer state_dict, epoch, best_acc
- [ ] Handle device mapping (CPU ↔ GPU)

### AC10: Code Quality
- [ ] All classes have docstrings (Chinese + English)
- [ ] Type hints for function signatures
- [ ] Assertions for tensor shape validation
- [ ] Example:
  ```python
  assert video.shape == (B, T, 3, 224, 224), f"Expected (B,T,3,224,224), got {video.shape}"
  ```

## Definition of Done
- All 5 module files created in `src/models/`:
  - `gnss_encoder.py`
  - `visual_encoder.py`
  - `fusion.py`
  - `tc_adaptformer.py`
  - `__init__.py` (exports main model)
- Model can be instantiated: `model = TCAdaptFormer(num_classes=11)`
- Forward pass works with dummy inputs (tested in Story 4)
- Trainable parameter count ≈ 2.1M
- Code includes comprehensive Chinese comments
- All 10 acceptance criteria verified

## Technical Notes
- **Avoid 3D Convolutions**: Use 2D ViT processing T frames independently
- **Memory Optimization**: Process frames in batch rather than loop if possible
- **Dimension Tracking**: Add assertions at each step to catch shape mismatches early
- **Reuse Existing Code**: Adapt `MBT/models/visual_model.py` structure where applicable

## Example Instantiation
```python
from src.models import TCAdaptFormer

model = TCAdaptFormer(
    num_classes=11,
    adapter_dim=64,
    adapter_rank=8,
    num_heads=12
)

# Print summary
model.summary()
# Expected output:
# Total params: 88.1M
# Trainable params: 2.1M (2.4%)
# Frozen params: 86M (97.6%)

# Forward pass
import torch
video = torch.randn(8, 5, 3, 224, 224)
gnss = torch.randn(8, 7)
logits = model(video, gnss)
assert logits.shape == (8, 11)
```

## Dependencies
- PyTorch >= 2.0
- timm >= 0.9.0 (for ViT-B16)
- Reuse `MBT/models/pet_modules.py` for AdaptFormer
