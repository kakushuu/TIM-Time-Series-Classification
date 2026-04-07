# Story 1: TC-AdaptFormer Architecture Design Document

## User Story
As a researcher, I want a detailed architecture design document that specifies the TC-AdaptFormer multimodal fusion model for agricultural activity recognition, so that the development team has clear implementation guidance.

## Context
- **Data**: 6272 aligned video-GNSS samples at 1Hz, 11 agricultural activity classes
- **GNSS Features**: 7-dimensional (经度, 纬度, 速度, 深度, 方向角, 间距, 类型)
- **Video**: RGB frames 224x224, extracted as JPG files
- **Model Choice**: TC-AdaptFormer (ViT-B16 + AdaptFormer + GNSS Cross-Attention)
- **Training Constraint**: Parameter-efficient (freeze backbone, train ~2M adapter params)

## Acceptance Criteria

### AC1: GNSS Feature Encoding Specification
- [ ] Document specifies GNSS 7-dim → MLP(7→128→768) → Query embeddings
- [ ] Mathematical formula provided: `Q_gnss = MLP(gnss_features) ∈ R^(B×768)`
- [ ] Activation function specified (e.g., ReLU, GELU)

### AC2: Visual Stream Architecture
- [ ] ViT-B16 backbone specification with frozen weights
- [ ] Input dimension: `(B, T, 3, 224, 224)` where T=5 frames
- [ ] Output patch tokens: `(B, T, 196, 768)` (14×14 patches)
- [ ] Clarify whether T frames are processed independently or with temporal encoding

### AC3: Cross-Attention Fusion Mechanism
- [ ] Formula: `Attn(Q=Q_gnss, K=visual_tokens, V=visual_tokens)`
- [ ] Specify attention head count (e.g., 12 heads)
- [ ] Output dimension: `(B, 768)` after attention pooling

### AC4: AdaptFormer Configuration
- [ ] Adapter rank specified (e.g., rank=8)
- [ ] Adapter dimension (e.g., 64)
- [ ] Placement: which transformer blocks get adapters (e.g., all 12 blocks)
- [ ] Trainable parameters estimate: ~2M

### AC5: Temporal Pooling Strategy
- [ ] Specify pooling method: mean/max/attention-weighted across T=5 frames
- [ ] Input: `(B, T, 768)` → Output: `(B, 768)`

### AC6: Classification Head
- [ ] Linear layer: `768 → 11` classes
- [ ] Loss function: CrossEntropyLoss with class weights (due to imbalance)
- [ ] Class weight calculation method documented

### AC7: Complete Forward Pass Dimensions
- [ ] End-to-end dimension flow table from input to output
- [ ] Example:
  ```
  Input Video: (8, 5, 3, 224, 224)
  Input GNSS: (8, 7)
  → ViT patches: (8, 5, 196, 768)
  → GNSS query: (8, 768)
  → Cross-attn: (8, 768)
  → Classifier: (8, 11)
  ```

### AC8: Parameter Count Breakdown
- [ ] Frozen ViT-B16: ~86M params (frozen)
- [ ] AdaptFormer adapters: ~2M params (trainable)
- [ ] GNSS MLP: ~0.1M params (trainable)
- [ ] Classifier head: ~8K params (trainable)
- [ ] Total trainable: ~2.1M params

## Definition of Done
- Architecture design document exists at `docs/tc_adaptformer_architecture.md`
- All 8 acceptance criteria are met with concrete specifications
- Document includes at least one architecture diagram (ASCII art or description)
- Mathematical formulas use LaTeX notation
- Dimension flow is verified to be consistent (no shape mismatches)

## Technical Notes
- Use existing MBT codebase as reference: `/home/research/Agri-MBT/Multimodal-Fusion-with-Attention-Bottlenecks-main/MBT/`
- Aligned data CSV: `/home/research/Agri-MBT/data/aligned_output/aligned_data.csv`
- Class distribution (highly imbalanced):
  - Class 3: 2304 samples
  - Class 7: 1444 samples
  - Class 1: 98 samples (minority)
