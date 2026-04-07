# IDEA_REPORT: Agricultural Activity Recognition from GPS Trajectories
## Idea Discovery Pipeline — Workflow 1

**Date:** 2026-03-16
**Author:** Claude Code (Workflow 1 — Automated Idea Discovery)
**Task:** 11-class agricultural activity recognition from GPS trajectory sequences + video frames

---

## 1. Executive Summary

**Top Recommendation: Fix the CLS Token Bug + Attention Pooling + Class-Weighted Loss (Idea A1+A2)**

The trajectory-only model is completely broken. It predicts class_7 for every single sample (42.47% accuracy = class_7 prior; macro F1 = 5.42; all other 10 classes have 0% recall). The loss flatlines at 1.76 from epoch 1 through epoch 15. The model learns nothing.

The root cause is a combination of two independently fatal bugs:
1. The CLS token is prepended to the BiLSTM outputs but has no mechanism to aggregate information from them — it is initialized to zeros and never receives gradient signal through the BiLSTM hidden states.
2. There is no class-weighted loss, so the optimizer immediately discovers that predicting class_7 for everything achieves ~42% accuracy with entropy ≈ log(1/0.388) ≈ 0.95 — lower than the entropy of the true uniform distribution log(11) ≈ 2.40 — and the gradient descent converges to this trivial solution.

**The single highest-leverage fix** is Idea A1 (replace `x[:, 0]` with attention pooling over all T BiLSTM hidden states) combined with Idea A2 (class-weighted cross-entropy loss with weights inversely proportional to class frequency). Together these can be implemented in under 30 minutes and should transform trajectory-only accuracy from 42% (trivial collapse) to a meaningful learning signal.

Expected outcome after fixing bugs: 55–70% trajectory-only accuracy based on comparisons with GAN-BiLSTM (92% F1 on binary field/road task) and ConvTEBiLSTM (97% F1 on binary task) — noting that our 11-class problem is harder.

**Secondary recommendation:** Idea B1 (PatchTST-style patch encoding for GPS sequences) as the architecture innovation that best addresses the T=512 long-sequence gradient problem.

---

## 2. Root Cause Analysis

### 2.1 Complete Trajectory-Only Collapse

Evidence from `experiments/results_trajectory_only.json`:
- Val accuracy: **42.469% every epoch** — identical to class_7 prior (38.8% of 32,249 samples)
- class_7: precision=42.47, recall=100.0 — model predicts ONLY this class
- All other 10 classes: precision=0, recall=0, F1=0
- Train loss: 1.778 → 1.758 over 15 epochs — gradient descent makes essentially no progress
- Loss plateau at ~1.76 corresponds to the entropy of predicting class_7 with probability 1.0 in an 11-class system with class_7 prior = 38.8%: -log(0.388) ≈ 0.946 averaged, which with CE including wrong classes gives ~1.76

### 2.2 Bug 1: CLS Token Disconnected from BiLSTM Hidden States

In `MBT/models/visual_model.py`, the `trajectory_only` forward pass:

```python
# In forward_traj_features():
rnn_out, _ = self.traj_bilstm(x)          # (bs, T, 768) — BiLSTM hidden states
cls = self.traj_cls_token.expand(B, -1, -1)   # (bs, 1, 768) — zero-initialized
x = torch.cat([cls, rnn_out], dim=1)           # (bs, T+1, 768)

# In forward():
x = self.forward_traj_features(x)       # (bs, T+1, 768)
x = x[:, 0]                              # BUG: takes ONLY the cls token
```

The CLS token at position 0 is `traj_cls_token`, a `nn.Parameter(torch.zeros(1, 1, 768))`. It is never updated by any attention mechanism — there is no cross-attention or self-attention over the sequence in trajectory_only mode. The BiLSTM outputs `rnn_out` (positions 1..T) are concatenated but then discarded entirely by `x[:, 0]`. The CLS token gradient flows only through `traj_encoder` → `classifier`, meaning it learns a global bias but has no connection to the actual trajectory content.

**This is equivalent to classifying every sequence with a random 768-dimensional vector that ignores all input data.**

### 2.3 Bug 2: No Class-Weighted Loss

With 11 classes, class_7 comprising 38.8% of data, and a model that outputs logits from a zero-initialized CLS token, the optimizer finds the trivial solution of pushing logit_7 high for all samples within the first batch. Without class weighting, the cross-entropy gradient of CE = -log(softmax(logit_7)) is minimized by this strategy. There is no penalty for ignoring minority classes.

The combination of Bug 1 (gradient path that ignores input) and Bug 2 (no class weighting) makes collapse inevitable.

### 2.4 Why Multimodal Works at 94.18%

The multimodal model succeeds because the ViT visual backbone is a pretrained, powerful encoder that provides strong representations. Vision dominates — the trajectory branch contributes marginally because: (a) the same CLS token bug exists in the multimodal path (though here `traj_post_norm` and the transformer encoder blocks provide some signal), and (b) the final prediction uses `(traj_cls + rgb_cls) * 0.5`. The vision signal is so strong that the average still produces 94% accuracy. However, this means the trajectory modality is contributing very little — the multimodal model is essentially a vision-only model.

### 2.5 Sequence Length Issue

T=512 with a 2-layer BiLSTM creates an effective gradient path of 512 × 2 = 1024 time steps. Even with BiLSTM's gating, information from the beginning of the sequence is substantially attenuated. This is an architectural bottleneck that limits what the model can learn even after fixing the CLS bug.

---

## 3. Literature Landscape

### 3.1 Agricultural Trajectory Classification

**[P1] GAN-BiLSTM: Field-Road Classification on Imbalanced GNSS Recordings**
Zhai et al. (2024). *Computers and Electronics in Agriculture*, 216, 108457.
https://www.sciencedirect.com/science/article/abs/pii/S0168169923008451
- Uses CTGAN to augment minority class trajectories to balance field/road distribution
- Att-BiLSTM with time-window feature augmentation achieves 92.3% accuracy, 92.1% F1
- Key insight: GAN-based data augmentation for trajectory imbalance outperforms SMOTE on sequential GPS data
- Relevance: Direct architectural inspiration for our trajectory branch; their GAN approach is the best published solution to GPS trajectory imbalance

**[P2] ConvTEBiLSTM: Fusing Local and Global Trajectory Features**
Chen et al. (2024). *MDPI IJGI*, 13(3), 90.
https://www.mdpi.com/2220-9964/13/3/90
- Combines 1D-Conv (local features), Transformer-Encoder (global features), BiLSTM (fusion) for GNSS classification
- Achieves 97.38% accuracy, 92.74% F1 on binary field/road task
- Key insight: Local+global feature fusion via Conv+Transformer is superior to BiLSTM alone
- Relevance: Architecture template — their Conv+TE+BiLSTM design directly applicable to our 27-feature sequence

**[P3] ATRNet: General Image Classification for Agricultural Trajectory Mode**
(2024). *Computers and Electronics in Agriculture*, 212, 108202.
https://www.sciencedirect.com/science/article/abs/pii/S0168169924010202
- Converts trajectory to image representation (25 features → 5×5 pixels), applies CNN
- Uses CTGAN for data balancing before classification
- Key insight: Trajectory-as-image avoids sequential gradient problems entirely; CNN-based spatial reasoning captures patterns unavailable to LSTM
- Relevance: Trajectory image representation is directly applicable (27 features → 5×5 or 6×6 image)

**[P4] STF+VFAU+BiLSTM: Multimodal Agricultural Trajectory Fusion**
Chen et al. (2023). *Computers and Electronics in Agriculture*.
- Maps trajectory points to pixel values, applies Attention U-Net for visual features, fuses with BiLSTM
- Key insight: 2D visual feature extraction from trajectory images provides richer spatial context
- Relevance: Architecture for converting trajectory sequences into spatial representations for attention-based processing

### 3.2 Time Series Classification with Transformers

**[P5] PatchTST: A Time Series is Worth 64 Words**
Nie et al. (2023). *ICLR 2023*.
https://arxiv.org/abs/2211.14730
- Segments time series into patches (sub-sequences) as Transformer tokens
- Channel-independent design: each variate processed separately, sharing Transformer weights
- Achieves 21% MSE reduction over prior Transformers; supports classification via `PatchTSTForClassification`
- Key insight: Patch-based tokenization captures local temporal patterns while reducing quadratic attention cost — critical for T=512 sequences
- Relevance: PatchTST with patch_size=16 and stride=8 would reduce 512 steps to ~32 tokens, eliminating long-sequence gradient issues

**[P6] iTransformer: Inverted Transformers Are Effective for Time Series Forecasting**
Liu et al. (2024). *ICLR 2024 Spotlight*.
https://arxiv.org/abs/2310.06625
- Inverts the tokenization axis: each variate (feature) becomes a token, attention captures inter-variate correlations
- FFN applied per token to learn temporal patterns; no positional embedding needed
- Key insight: For multivariate time series with inter-correlated features (e.g., speed correlates with angular_speed), variate-level attention captures feature interactions that token-level attention misses
- Relevance: Our 27 features have strong inter-correlations (speed, accel, angular_speed all derived from lat/lon); iTransformer's variate-level attention would naturally capture these

**[P7] FormerTime: Hierarchical Multi-Scale Representations for Time Series Classification**
Cheng et al. (2023). *ACM Web Conference 2023*, pp. 1437–1445.
https://arxiv.org/pdf/2302.09818
- Hierarchical Transformer with temporal slice partition for multi-scale representation
- Captures both local and global temporal dependencies via hierarchical attention
- Key insight: Multi-scale temporal features (short-term motion patterns + long-term trajectory shape) are crucial for activity recognition in GPS sequences
- Relevance: T=512 contains activities at multiple timescales — turning maneuvers (10-50 steps) and overall field patterns (100-500 steps)

### 3.3 Class Imbalance in Sequence Classification

**[P8] T-SMOTE: Temporal-Oriented Synthetic Minority Oversampling for Imbalanced Time Series**
Zhao et al. (2022). *IJCAI 2022*.
https://www.ijcai.org/proceedings/2022/334
- Generates synthetic minority-class time series samples near the decision boundary while preserving temporal structure
- Key insight: Standard SMOTE corrupts temporal correlations — T-SMOTE interpolates in the temporal domain to maintain sequential patterns
- Relevance: For oversampling rare classes (class_1: 393 samples) in our 512-step trajectory sequences, T-SMOTE preserves the GPS motion patterns that define the activity

**[P9] Batch-Balanced Focal Loss (BBFL)**
(2023). *PMC/SPIE*.
https://pmc.ncbi.nlm.nih.gov/articles/PMC10289178/
- Combines batch-balancing (equal class representation per batch) with focal loss
- Addresses both class-frequency imbalance (batch balancing) and sample-difficulty imbalance (focal loss)
- Key insight: Pure focal loss helps with hard samples but not with systematic class-level imbalance; combining with batch balancing is more robust
- Relevance: Our problem has both: systematic class imbalance (38.8% class_7) and hard samples within each class (GPS noise, weather, driver habit variation)

### 3.4 Multimodal Fusion

**[P10] Attention Bottlenecks for Multimodal Fusion (MBT)**
Nagrani et al. (2021/NeurIPS 2021).
https://arxiv.org/abs/2107.00135
- The foundational paper this project adapts; introduces bottleneck latent tokens for cross-modal attention
- 50% FLOPs reduction vs. full multimodal attention; state-of-the-art on AudioSet, VGGSound
- Key insight: Cross-modal bottleneck tokens force compression of each modality before sharing, enabling each stream to specialize in early layers
- Relevance: Core architecture; the bottleneck design is appropriate but the trajectory stream needs to produce meaningful tokens first (Bug 1 must be fixed)

**[P11] Multi-Modal Representation via Contrastive Learning with Attention Bottleneck Fusion**
(2023). *PMC*.
https://pmc.ncbi.nlm.nih.gov/articles/PMC10606612/
- Combines MBT-style bottleneck fusion with contrastive loss between modalities
- Attentive Statistics Fusion captures long-term fluctuations across modalities
- Key insight: Adding contrastive loss between trajectory and video representations ensures the two modalities develop complementary (not redundant) representations
- Relevance: If trajectory branch is fixed and multimodal model is already at 94%, contrastive loss could push accuracy higher while ensuring trajectory provides unique signal

**[P12] FACNet: Deep Classification of Frequently-Changing Activities from GPS Trajectories**
(ACM SIGSPATIAL 2022).
https://www.amazon.science/publications/deep-classification-of-frequently-changing-activities-from-gps-trajectories
- BiLSTM + custom attention for GPS-only activity classification with frequent mode changes
- Infers modality of GPS points in trajectory without additional inputs
- Key insight: Attention over BiLSTM hidden states (not CLS token) is the standard design in GPS classification literature — confirming that our Bug 1 is a known anti-pattern
- Relevance: Direct evidence that the correct approach for GPS sequence classification is attention-pooled BiLSTM outputs, not CLS token

---

## 4. All Ideas

### Idea A1: Fix CLS Bug — Replace with Attention Pooling
**Problem addressed:** Bug 1 (CLS token disconnected from BiLSTM outputs)
**Method:**
Replace `x = x[:, 0]` with learned attention pooling:
```python
# In forward() for trajectory_only:
x = self.forward_traj_features(x)    # (bs, T+1, 768)
x = x[:, 1:]                          # (bs, T, 768) — BiLSTM hidden states only
scores = self.traj_attn_w(x)          # (bs, T, 1) — already defined in __init__
scores = torch.softmax(scores, dim=1) # (bs, T, 1)
x = (scores * x).sum(dim=1)           # (bs, 768) — weighted sum of all hidden states
x = self.traj_encoder(x)              # (bs, 256)
logits = self.classifier(x)
```
Note: `self.traj_attn_w = nn.Linear(768, 1)` already exists in `__init__` — it just needs to be used.
**Expected improvement:** From 42% (random collapse) to 55-70% by providing the model with actual trajectory information
**Implementation effort:** Low (15 minutes — 3 lines of code changed)
**Novelty:** This is the standard design in GPS classification literature ([P12]) — we are fixing a bug, not inventing something new
**Pilot experiment:** Change 3 lines, retrain for 5 epochs with class-weighted loss, check if loss actually decreases and all 11 classes are predicted

### Idea A2: Class-Weighted Cross-Entropy Loss
**Problem addressed:** Bug 2 (no class weighting → majority class collapse)
**Method:**
Compute class weights in `train_test.py` before creating the loss:
```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Compute inverse-frequency weights
classes = np.arange(num_classes)
weights = compute_class_weight('balanced', classes=classes, y=train_labels)
class_weights = torch.FloatTensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```
Alternatively, effective number weighting (He et al., 2019): `w_i = (1 - beta) / (1 - beta^n_i)` with beta=0.9999.
**Expected improvement:** Prevents trivial collapse; forces the model to learn representations for minority classes (class_1: 393 samples → weight ~82× majority)
**Implementation effort:** Low (10 minutes)
**Novelty:** Standard technique; critical bug fix
**Pilot experiment:** Same as A1 — always combine A1+A2 as a single "bug fix" experiment

### Idea A3: Focal Loss for Dynamic Class-Difficulty Weighting
**Problem addressed:** Bug 2 + hard sample learning
**Method:**
Replace CE with focal loss: `FL(p_t) = -(1 - p_t)^gamma * log(p_t)` with `gamma=2.0`.
Optionally combine with class weights: `FL_w(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)`.
Reference implementation: `torchvision.ops.sigmoid_focal_loss` or custom CE-based implementation.
**Expected improvement:** Better than pure class-weighted CE for samples near decision boundaries; especially useful for class_1 (393 samples, likely hard examples due to rarity)
**Implementation effort:** Low (20 minutes)
**Novelty:** Combining focal loss with trajectory sequence classification on GPS data is not well-explored
**Pilot experiment:** Train trajectory_only with A1 + focal loss (gamma=2, alpha=class_weights) for 10 epochs, compare macro-F1 with A1+A2

### Idea B1: PatchTST-Style Patch Encoding for GPS Sequences
**Problem addressed:** T=512 long-sequence gradient vanishing; lack of local temporal context
**Method:**
Replace BiLSTM with a patch-based Transformer encoder:
1. Split 512-step sequence into non-overlapping patches of size P=16 → 32 tokens
2. Project each patch: `Linear(27*16, 768)` → 32 tokens of dim 768
3. Add learnable positional embeddings for 32 patch positions
4. Apply 4–6 standard Transformer encoder blocks (MSA + FFN)
5. Mean-pool over 32 tokens (or use CLS token with proper cross-attention)

Architecture sketch:
```
Input: (bs, 512, 27)
→ Reshape: (bs, 32, 16*27) = (bs, 32, 432)
→ Linear(432, 768): (bs, 32, 768)
→ + pos_embed(32, 768)
→ TransformerEncoder(6 layers, 8 heads, dim=768)
→ Mean pool: (bs, 768)
→ MLP classifier: (bs, 11)
```
**Expected improvement:** Reduces effective sequence length from 512 to 32 (16× reduction in attention cost), enables each token to capture 16-step local patterns (e.g., a turning maneuver), and allows attention to model relationships between distant temporal patches
**Implementation effort:** Medium (2–3 hours — implement PatchEncoder class, integrate into existing model)
**Novelty:** PatchTST (ICLR 2023) applied to GPS trajectory classification for agricultural activity recognition — no published work combines PatchTST with agricultural GPS trajectories
**Pilot experiment:** Implement PatchEncoder, train trajectory_only with patch sizes {8, 16, 32} and class-weighted loss, compare macro-F1 with BiLSTM baseline (after fixing bugs)

### Idea B2: Hierarchical Dual-Scale BiLSTM
**Problem addressed:** T=512 long-sequence gradient; multi-scale temporal patterns
**Method:**
Two-level hierarchical LSTM:
- Level 1 (local): BiLSTM over windows of 32 steps → 16 local context vectors (512/32=16)
- Level 2 (global): BiLSTM over the 16 context vectors → final sequence representation

```
Input: (bs, 512, 27)
→ Reshape: (bs, 16, 32, 27)
→ Local BiLSTM (per window): (bs, 16, 256)   # shared weights
→ Global BiLSTM: (bs, 16, 256)
→ Attention pool: (bs, 256)
→ MLP classifier: (bs, 11)
```
**Expected improvement:** Reduces effective gradient path from 512 to 32 (local) + 16 (global) = 48 steps; each level captures patterns at its natural timescale (individual maneuvers vs. overall activity structure)
**Implementation effort:** Medium (2 hours)
**Novelty:** Hierarchical BiLSTM for 11-class agricultural machinery activity recognition; directly inspired by [P7] FormerTime but using LSTMs instead of Transformers for efficiency
**Pilot experiment:** Implement HierarchicalBiLSTM class, train with A2 loss, compare with flat BiLSTM on trajectory_only task

### Idea B3: iTransformer for Feature-Correlated GPS Sequences
**Problem addressed:** Under-exploitation of inter-feature correlations in 27-feature GPS data
**Method:**
Apply the iTransformer approach ([P6]): instead of treating each time step as a token, treat each of the 27 features as a token with a T=512 time series as its value.

```
Input: (bs, 512, 27)
→ Transpose: (bs, 27, 512)
→ For each feature: Linear(512, 768) → feature token
→ Self-attention over 27 feature tokens (captures lat-lon-speed correlations)
→ FFN per token (learns temporal patterns within each feature)
→ Mean pool over 27 tokens: (bs, 768)
→ MLP classifier: (bs, 11)
```
**Expected improvement:** The 27 features are highly correlated (speed, angular_speed, and their window statistics are all derived from lat/lon). iTransformer's variate-level attention would naturally discover that "high angular_speed + low speed = turning maneuver" combinations define specific activities.
**Implementation effort:** Medium (3 hours)
**Novelty:** iTransformer (ICLR 2024 Spotlight) has not been applied to GPS agricultural activity recognition; the agricultural domain has unique feature correlation patterns (field-specific motion signatures)
**Pilot experiment:** Implement iTransformerEncoder (variate tokenization), train trajectory_only with A2 loss, compare macro-F1 and per-class recall with PatchTST baseline

### Idea C1: Trajectory-Guided Cross-Modal Attention in MBT
**Problem addressed:** Vision dominates multimodal model (trajectory contributes little); trajectory branch has no mechanism to direct visual attention
**Method:**
Modify the MBT fusion to use trajectory features as queries for visual attention. In the AdaptFormer blocks, instead of symmetric bottleneck tokens shared between both modalities, create asymmetric attention:
- Trajectory tokens attend to visual tokens as keys/values (trajectory queries visual context)
- Visual tokens also attend to trajectory tokens (visual queries trajectory context)
- Bottleneck tokens become modality-asymmetric: some attend only in the trajectory→visual direction

Concrete change in `pet_modules.py`:
```python
# In AdaptFormer forward:
# Standard: both modalities share bottleneck tokens
# Modified: trajectory features explicitly attend to visual positions
traj_to_visual = cross_attention(Q=traj_tokens, K=visual_tokens, V=visual_tokens)
visual_from_traj = cross_attention(Q=visual_tokens, K=traj_tokens, V=traj_tokens)
```
**Expected improvement:** Allows the model to ask "where in the video is the activity corresponding to this trajectory pattern?" — which is precisely what an agricultural inspector would do. Should improve minority class accuracy where visual appearance is ambiguous but trajectory is distinctive (or vice versa).
**Implementation effort:** High (4–6 hours, requires modifying AdaptFormer architecture)
**Novelty:** Trajectory-guided visual attention in MBT for agricultural activity recognition — no published work applies this pattern to GPS+video fusion in agriculture
**Pilot experiment:** Modify AdaptFormer to add one trajectory→visual cross-attention layer (in the last 4 of 12 blocks), compare multimodal accuracy and per-class F1 with baseline

### Idea C2: Cross-Modal Contrastive Regularization Loss
**Problem addressed:** Vision dominates (trajectory branch is redundant); no explicit pressure for trajectory to contribute unique information
**Method:**
Add a contrastive loss term that pulls same-class video and trajectory representations together while pushing different-class pairs apart. Inspired by [P11].

```python
# In train_test.py:
# Standard CE:
ce_loss = criterion(logits, labels)

# Contrastive term (InfoNCE-style, per [P11]):
traj_embed = normalize(traj_cls)   # (bs, 768)
vis_embed = normalize(rgb_cls)     # (bs, 768)
sim_matrix = traj_embed @ vis_embed.T / temperature
# Positive pairs: same sample (diagonal)
# Negative pairs: different samples in batch
contrastive_loss = -log_softmax(sim_matrix, dim=1).diag().mean()

total_loss = ce_loss + lambda_c * contrastive_loss   # lambda_c = 0.1
```
**Expected improvement:** Forces the trajectory embedding to be semantically aligned with the video embedding for the same sample. This regularization prevents the trajectory branch from collapsing while the video branch does all the work.
**Implementation effort:** Low–Medium (1–2 hours)
**Novelty:** Cross-modal contrastive regularization between GPS trajectory and video representations in agricultural activity recognition — the trajectory+video combination is unique in this domain
**Pilot experiment:** Train multimodal model with added contrastive term (lambda=0.1), compare: (a) overall accuracy, (b) per-class F1 for minority classes, (c) trajectory-only accuracy (test trajectory branch alone after multimodal training)

### Idea D1: Trajectory-as-Image (Velocity/Acceleration Heatmap)
**Problem addressed:** Sequential gradient issues; inability to capture spatial patterns in GPS traces
**Method:**
Inspired by [P3] (ATRNet) and [P4] (STF+VFAU+BiLSTM): convert the 512-step trajectory sequence into a 2D image representation and apply CNN + ViT.

Two complementary representations:
1. **Spatial map:** Plot (lat, lon) as a 2D raster image (224×224), encode speed as pixel intensity and angular_speed as color channel. Apply ViT or ResNet.
2. **Feature map:** Reshape 27 features × 512 steps into a 2D array (27×512 → resize to 224×224), treat as a single-channel image capturing temporal evolution of all features simultaneously.

The feature-map approach is simpler and directly applicable:
```python
# Input: (bs, 512, 27)
# → Transpose: (bs, 27, 512)
# → Interpolate to (bs, 27, 512) → (bs, 1, 224, 224)  [using 2D bilinear resize]
# → ViT or CNN encoder
```
**Expected improvement:** CNN feature extraction is well-suited to 2D spatial patterns. The temporal evolution of 27 features plotted as a 2D image exposes horizontal (temporal) and vertical (feature-type) patterns invisible to 1D temporal models. Activities likely have distinctive "texture" signatures in this representation.
**Implementation effort:** Low–Medium (2 hours)
**Novelty:** Applying 2D image-based representation of GPS kinematic features for 11-class agricultural activity recognition — extends [P3]'s binary classification approach to multi-class with ViT backbone
**Pilot experiment:** Preprocess all trajectories to 27×512 → 224×224 images (one-time preprocessing), train frozen ViT-B16 classifier (same backbone as multimodal model), compare with BiLSTM baseline on trajectory_only

### Idea D2: Multi-Scale Window Feature Augmentation
**Problem addressed:** Fixed T=512 sequence misses patterns at different timescales; under-exploitation of window statistics already in features
**Method:**
Our data already includes 5-window and 50-window statistics. Extend this to learn features at multiple scales explicitly:
1. Extract trajectory subsequences at 3 timescales: last 64, last 128, last 512 GPS points
2. Process each scale with a separate BiLSTM + attention pool → 3 × 768-dim embeddings
3. Concatenate and project: Linear(3×768, 768) → MLP classifier

```python
# Multi-scale encoder:
x_short = x[:, -64:, :]    # last 64 steps (local)
x_mid   = x[:, -128:, :]   # last 128 steps (medium)
x_full  = x                 # all 512 steps (global)

h_short = attn_pool(bilstm(x_short))   # (bs, 768)
h_mid   = attn_pool(bilstm(x_mid))     # (bs, 768) — shared LSTM weights
h_full  = attn_pool(bilstm(x_full))    # (bs, 768)

h = concat([h_short, h_mid, h_full])   # (bs, 2304)
h = linear(2304→768)                   # (bs, 768)
logits = classifier(h)                 # (bs, 11)
```
**Expected improvement:** Different activities have distinctive patterns at different timescales — a sharp turn (class distinction) may be visible in 64 steps but diluted in 512; an overall field coverage pattern requires 512 steps
**Implementation effort:** Medium (3 hours)
**Novelty:** Multi-scale trajectory encoding for agricultural activity recognition; extends the window statistics already in the data to full representation learning
**Pilot experiment:** Implement MultiScaleBiLSTM, train with A2 loss, compare per-class F1 with single-scale BiLSTM baseline

---

## 5. Ranked Top 3 Ideas

### Ranking Criteria

| Idea | Expected Impact | Feasibility | Novelty | Risk | Total Score |
|------|----------------|-------------|---------|------|-------------|
| A1+A2 (CLS fix + class weight) | 9/10 | 10/10 | 2/10 | 1/10 (very low) | 22/40 |
| B1 (PatchTST) | 7/10 | 7/10 | 8/10 | 3/10 | 25/40 |
| C2 (Contrastive) | 6/10 | 8/10 | 7/10 | 4/10 | 25/40 |
| B2 (Hierarchical BiLSTM) | 6/10 | 8/10 | 5/10 | 3/10 | 22/40 |
| C1 (Traj-guided attention) | 7/10 | 4/10 | 9/10 | 6/10 | 26/40 |
| D1 (Trajectory-as-image) | 6/10 | 7/10 | 6/10 | 5/10 | 24/40 |
| B3 (iTransformer) | 5/10 | 7/10 | 8/10 | 5/10 | 25/40 |

### Rank 1: Idea A1+A2 — Fix CLS Bug + Class-Weighted Loss

**Why #1:** Maximum expected impact for minimum effort. This is not an incremental improvement — it is the difference between a working model and a broken model. Without this fix, no other idea can be evaluated. All downstream experiments depend on a functioning trajectory branch.

**Pilot Experiment Design:**
```
Experiment: traj_baseline_fixed
Changes:
  1. In visual_model.py forward() trajectory_only branch:
     - Change: x = x[:, 0]
     - To:     x = x[:, 1:]  # use BiLSTM hidden states
               scores = F.softmax(self.traj_attn_w(x), dim=1)
               x = (scores * x).sum(dim=1)

  2. In train_test.py, replace criterion:
     - Compute class_weights from training set
     - criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

Training: 15 epochs, lr=3e-4, batch=8 (same as baseline)
Metrics to track:
  - Loss curve (must actually decrease)
  - Per-class recall (must show >0 for all 11 classes)
  - Macro-F1 (target: >25%, compared to 5.42% baseline)
  - Best val accuracy (target: >55%)
```

**Success criteria:** Loss decreases in epoch 1 (not flatlines); at least 8 of 11 classes have recall > 10%.

### Rank 2: Idea B1 — PatchTST-Style Patch Encoding

**Why #2:** After fixing Bug 1+2, the next bottleneck is the BiLSTM's difficulty with T=512 long sequences. PatchTST-style patching is the state-of-the-art solution (ICLR 2023) and has a clean implementation path. The approach is well-validated on diverse time series benchmarks and directly applicable here.

**Pilot Experiment Design:**
```
Experiment: traj_patchtst
Architecture:
  class PatchEncoder(nn.Module):
    patch_size = 16
    n_patches = 512 // 16 = 32
    proj = nn.Linear(27 * 16, 768)  # project each patch
    pos_embed = nn.Parameter(torch.randn(1, 32, 768))
    encoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=2048, dropout=0.1),
        num_layers=4
    )

  Forward:
    x: (bs, 512, 27)
    → x.reshape(bs, 32, 16*27)  # (bs, 32, 432)
    → proj(x) + pos_embed       # (bs, 32, 768)
    → encoder                   # (bs, 32, 768)
    → x.mean(dim=1)             # (bs, 768) — mean pooling
    → traj_encoder(x)           # (bs, 256) — existing MLP
    → classifier(x)             # (bs, 11)

Training: 15 epochs, lr=3e-4, batch=8, class-weighted loss
Compare: macro-F1 vs. BiLSTM+attn_pool (Rank 1 baseline)
Ablation: patch sizes {8, 16, 32}; pooling {mean, attention, CLS}
```

**Success criteria:** Macro-F1 > BiLSTM+attn_pool baseline from Rank 1 experiment by >5 points.

### Rank 3: Idea C2 — Cross-Modal Contrastive Regularization

**Why #3:** The multimodal model achieves 94.18% but the trajectory branch likely contributes very little signal. Contrastive regularization is the most principled way to force the trajectory branch to encode complementary information. It has low implementation risk (additive loss term, no architectural changes), is grounded in established contrastive learning theory ([P11]), and directly addresses the "vision dominance" problem. If successful, it could push multimodal accuracy above 95% while also improving the trajectory branch's standalone quality.

**Pilot Experiment Design:**
```
Experiment: multimodal_contrastive
Changes to train_test.py:
  In training loop:
    logits, traj_cls, rgb_cls = model(x, y, return_embeddings=True)
    ce_loss = criterion(logits, labels)

    # InfoNCE contrastive loss
    traj_norm = F.normalize(traj_cls, dim=-1)  # (bs, 768)
    vis_norm  = F.normalize(rgb_cls, dim=-1)   # (bs, 768)
    sim = traj_norm @ vis_norm.T / 0.07        # (bs, bs)
    targets = torch.arange(bs, device=device)
    contra_loss = (F.cross_entropy(sim, targets) +
                   F.cross_entropy(sim.T, targets)) / 2

    loss = ce_loss + 0.1 * contra_loss

Changes to visual_model.py:
  In multimodal forward(), return traj_cls and rgb_cls before averaging

Training: 15 epochs, lr=3e-4, batch=8, class-weighted loss
Metrics:
  - Overall val accuracy (baseline: 94.18%)
  - class_1 F1 (baseline: 49.69% — hardest class)
  - class_2 F1 (baseline: 63.26% — second hardest)
  - Trajectory-only accuracy when tested standalone after multimodal training

Lambda sweep: {0.01, 0.1, 0.5} for contrastive weight
```

**Success criteria:** Overall accuracy maintained (>93%) while class_1 F1 improves by >5 points.

---

## 6. Critical Review of Top 3

### 6.1 Critical Review of A1+A2 (Fix CLS Bug + Class Weighting)

**Weaknesses:**
1. Even with attention pooling, 2-layer BiLSTM on T=512 may still have gradient attenuation for the beginning of sequences. The fix is necessary but may not be sufficient for good accuracy.
2. Class-weighted CE with extreme weights (class_1 weight ~82×) can cause training instability — the gradient from a single class_1 misclassification could dominate an entire batch.
3. Attention pooling without positional awareness may struggle to distinguish activities that have similar feature distributions but different temporal structures.

**Validation experiments needed:**
- Gradient norm monitoring per epoch (check for instability from extreme class weights)
- Compare inverse-frequency weighting vs. effective number weighting (He et al., 2019)
- Ablation: attention pooling vs. mean pooling vs. last-hidden-state (BiLSTM's final state)
- Test on balanced subset first: sample equal number per class, train without weighting, check if model can learn all classes with sufficient data

**What could go wrong:**
- Training instability with extreme class weights → solution: clip gradients, reduce learning rate, use effective number weighting instead of pure inverse frequency
- Attention pooling learns degenerate solution (always attends to one or two positions) → solution: add attention entropy regularization or use multi-head attention pooling
- Even fixed BiLSTM may only reach 50–55% on 11-class problem (inherently harder than 2-class field/road in published work) → set realistic expectations

### 6.2 Critical Review of B1 (PatchTST-Style Patch Encoding)

**Weaknesses:**
1. GPS sequences may have variable-density events (agricultural machinery often has long idle periods) — fixed-size patching treats idle and active periods equally.
2. Patch size choice is a hyperparameter with no principled selection method for this specific domain; wrong patch size could hurt performance (e.g., if key events span exactly the patch boundary).
3. Training a Transformer from scratch on 32,249 sequences (vs. fine-tuning a pretrained model) is data-hungry; the dataset may be too small for good generalization.
4. The channel-independence assumption in PatchTST may be sub-optimal for our 27 highly-correlated features.

**Validation experiments needed:**
- Patch size sensitivity analysis: {4, 8, 16, 32, 64}
- Compare random initialization vs. self-supervised pretraining on unlabeled trajectory data
- Verify that T=512 sequences have diverse content (not mostly idle/uniform motion)
- Test both channel-independent and channel-dependent versions

**What could go wrong:**
- Insufficient data for training 4-6 layer Transformer from scratch → solution: use smaller Transformer (2 layers), add strong regularization, or consider transfer learning from pretrained time series models
- Patch boundaries cut through important events → solution: use overlapping patches with stride < patch_size (e.g., stride=8 for patch_size=16)

### 6.3 Critical Review of C2 (Cross-Modal Contrastive Loss)

**Weaknesses:**
1. The contrastive loss assumes that matching video and trajectory from the same sample should be similar in embedding space — but for the same activity class, different samples may have very different appearance and trajectory due to varying lighting, field conditions, and machinery. This could create conflicting gradients.
2. Batch size 8 is very small for contrastive learning — InfoNCE loss quality degrades with few negative pairs. Standard contrastive learning uses batch sizes of 256–4096.
3. The temperature parameter (0.07) is crucial and sensitive; wrong temperature causes either gradient collapse (too low) or noisy gradients (too high).
4. If the trajectory branch is fundamentally noisy/uninformative (even after fixing Bug 1), contrastive loss may hurt the visual branch by pulling it toward bad trajectory embeddings.

**Validation experiments needed:**
- Experiment with larger effective batch size via gradient accumulation (effective batch = 64 by accumulating 8 steps)
- Temperature sweep: {0.05, 0.07, 0.1, 0.2}
- Monitor contrastive loss value separately from CE loss — should decrease gradually
- Ablation: contrastive loss added to multimodal model ONLY after trajectory branch is independently validated (run A1+A2 first)
- Check whether trajectory-only performance improves when fine-tuned from multimodal checkpoint

**What could go wrong:**
- Batch size 8 is too small → contrastive loss has low variance and provides weak signal → solution: use MoCo-style memory bank to accumulate embeddings across batches
- Visual branch degrades due to conflicting gradients → solution: use detach on visual branch when computing contrastive loss (only update trajectory branch with contrastive gradient)

---

## 7. Implementation Roadmap

### Day 1: Fix the Bugs (A1 + A2) — 1–2 hours

**Files to modify:**
1. `/home/research/Agri-MBT/Multimodal-Fusion-with-Attention-Bottlenecks-main/MBT/models/visual_model.py`
   - In `forward()`, `trajectory_only` branch: replace `x = x[:, 0]` with attention pooling over `x[:, 1:]`
   - The `self.traj_attn_w` linear layer already exists — just use it

2. `/home/research/Agri-MBT/Multimodal-Fusion-with-Attention-Bottlenecks-main/MBT/train_test.py`
   - After splitting train/val: compute class weights from training labels
   - Pass weights to `nn.CrossEntropyLoss(weight=class_weights)`

**Success check:** Run 2 epochs. Loss should decrease (not flatline). At least 5 of 11 classes should have non-zero predictions. Macro-F1 should be > 10%.

### Day 1 Afternoon: Validate Baseline + Ablate Loss Functions (A2 vs. A3)

- Run full 15 epochs with A1+A2 to establish a true baseline
- Run with A1+A3 (focal loss) to compare
- Document: per-class recall, macro-F1, loss curves

**Expected outcome:** trajectory_only macro-F1 rises from 5.42% to 30–50%.

### Day 2: Architecture Innovation — PatchTST Encoding (B1)

**File to create:**
`/home/research/Agri-MBT/Multimodal-Fusion-with-Attention-Bottlenecks-main/MBT/models/patch_encoder.py`
- Implement `PatchEncoder` class as described in Idea B1
- Integrate into `visual_model.py` as an alternative `traj_encoder`

**File to modify:**
- `train_test.py`: add `--traj_arch {bilstm, patchtst}` argument

**Expected outcome:** If PatchTST > BiLSTM+attn_pool by >5 macro-F1 points, adopt as trajectory encoder.

### Day 2 Afternoon: Contrastive Regularization (C2) on Multimodal

- Modify multimodal forward to return `(traj_cls, rgb_cls)` alongside logits
- Add InfoNCE contrastive loss term in training loop
- Test lambda={0.1, 0.5} and temperature={0.07, 0.1}

**Expected outcome:** Multimodal accuracy maintained (>93.5%); class_1 F1 should improve from 49.69% to >55%.

### Day 3: Trajectory-as-Image (D1) — Exploratory

- Preprocessing: write a script to convert (bs, 512, 27) → (bs, 224, 224) feature maps
- Store preprocessed tensors as .pt files for fast loading
- Train frozen ViT classifier on trajectory images

**Expected outcome:** This is a exploratory experiment. If ViT on trajectory images achieves >60% macro-F1, it opens a path to reuse the existing ViT backbone for trajectory encoding in the multimodal model.

### Final Integration

After validating individual ideas:
1. If B1 (PatchTST) > BiLSTM: replace trajectory encoder in both trajectory_only and multimodal models
2. If C2 (Contrastive) helps: add as permanent regularization in multimodal training
3. Combine best trajectory encoder (B1 or BiLSTM+attn) with best loss (A2 or A3) and best multimodal objective (CE + contrastive)

---

## 8. Next Steps toward Workflow 2 (Auto-Review Loop)

Workflow 2 should automatically:

1. **Run pilot experiments** for each ranked idea and record results to JSON
2. **Parse results** and update per-class F1, macro-F1, and loss curves
3. **Generate a review** comparing pilot results against predictions from this report
4. **Revise rankings** based on empirical results

### Specific Workflow 2 Inputs

After completing Day 1–3 experiments, the auto-review loop should collect:
- `experiments/results_traj_fixed.json` (A1+A2 results)
- `experiments/results_traj_patchtst.json` (B1 results)
- `experiments/results_multimodal_contrastive.json` (C2 results)
- `experiments/results_traj_image.json` (D1 results)

### Key Questions for Auto-Review

1. Does attention pooling (A1) actually fix the collapse? (Check: does class_1 recall > 0?)
2. Is the improvement from PatchTST (B1) over BiLSTM+attn_pool statistically meaningful?
3. Does contrastive loss (C2) help minority class performance without hurting overall accuracy?
4. Is the trajectory-as-image approach (D1) competitive with temporal sequence models?
5. What is the macro-F1 gap between trajectory_only (after fixes) and multimodal? This gap reveals how much unique information trajectory contributes beyond vision.

### Paper Narrative

The expected paper narrative, pending experimental results:

- **Problem:** Agricultural activity recognition from GPS + video is dominated by vision; GPS trajectory contribution is unclear
- **Finding:** Standard BiLSTM+CLS design is fundamentally broken for trajectory classification (quantify gap: 42% vs. X% after fix)
- **Contribution 1:** Attention-pooled BiLSTM as correct trajectory encoder for GPS sequences (A1)
- **Contribution 2:** PatchTST-style encoding for long GPS sequences (B1) — novel application domain
- **Contribution 3:** Cross-modal contrastive regularization to ensure trajectory contributes unique signal (C2)
- **Result:** X% trajectory-only accuracy (vs. 42% broken baseline), Y% multimodal accuracy (vs. 94.18% vision-dominated baseline), with improved minority class F1

---

## References (Key Papers)

1. [GAN-BiLSTM] Zhai et al. (2024). GAN-BiLSTM network for field-road classification on imbalanced GNSS recordings. *Computers and Electronics in Agriculture*, 216. https://www.sciencedirect.com/science/article/abs/pii/S0168169923008451

2. [ConvTEBiLSTM] Chen et al. (2024). ConvTEBiLSTM: A Neural Network Fusing Local and Global Trajectory Features for Field-Road Mode Classification. *MDPI IJGI*, 13(3), 90. https://www.mdpi.com/2220-9964/13/3/90

3. [ATRNet] (2024). A general image classification model for agricultural machinery trajectory mode recognition. *Computers and Electronics in Agriculture*. https://www.sciencedirect.com/science/article/abs/pii/S0168169924010202

4. [PatchTST] Nie et al. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. *ICLR 2023*. https://arxiv.org/abs/2211.14730

5. [iTransformer] Liu et al. (2024). iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. *ICLR 2024 Spotlight*. https://arxiv.org/abs/2310.06625

6. [FormerTime] Cheng et al. (2023). FormerTime: Hierarchical Multi-Scale Representations for Time Series Classification. *ACM Web Conference 2023*. https://arxiv.org/pdf/2302.09818

7. [T-SMOTE] Zhao et al. (2022). T-SMOTE: Temporal-oriented Synthetic Minority Oversampling Technique for Imbalanced Time Series Classification. *IJCAI 2022*. https://www.ijcai.org/proceedings/2022/334

8. [BBFL] (2023). Batch-balanced focal loss: a hybrid solution to class imbalance in deep learning. *PMC/SPIE*. https://pmc.ncbi.nlm.nih.gov/articles/PMC10289178/

9. [MBT] Nagrani et al. (2021). Attention Bottlenecks for Multimodal Fusion. *NeurIPS 2021*. https://arxiv.org/abs/2107.00135

10. [MBT-CL] (2023). Multi-Modal Representation via Contrastive Learning with Attention Bottleneck Fusion. *PMC*. https://pmc.ncbi.nlm.nih.gov/articles/PMC10606612/

11. [FACNet] (2022). Deep Classification of Frequently-Changing Activities from GPS Trajectories. *ACM SIGSPATIAL IWCTS 2022*. https://www.amazon.science/publications/deep-classification-of-frequently-changing-activities-from-gps-trajectories

12. [MobilityDL] (2024). MobilityDL: a review of deep learning from trajectory data. *GeoInformatica*. https://link.springer.com/article/10.1007/s10707-024-00518-8
