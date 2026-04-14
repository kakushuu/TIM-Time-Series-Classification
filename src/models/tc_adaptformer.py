"""
TAIF 风格的 TC-AdaptFormer 完整模型

升级后的结构：
  video (B,T,3,224,224) -> VisualEncoder -> (B,T,196,768)
  gnss  (B,T,F)         -> TrajectoryPatternEncoder -> traj_tokens, traj_global
                                 ↓
                     TemporalAlignmentFusion (TAM + IBF)
                                 ↓
                           fused feature (B,768)
                                 ↓
                           classifier (B,11)

为兼容旧代码，模型类名仍保留为 TCAdaptFormer。
"""

import torch
import torch.nn as nn

from .fusion import TemporalAlignmentFusion
from .gnss_encoder import TrajectoryPatternEncoder
from .visual_encoder import VisualEncoder


class TCAdaptFormer(nn.Module):
    """
    面向 Agricultural Machinery Trajectory Time-Series Classification 的
    多模态时间对齐与鲁棒融合模型。

    Input:
      video: (B, T, 3, 224, 224)
      gnss:  (B, T, F) 或 (B, F)

    Output:
      logits: (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int = 11,
        gnss_input_dim: int = 7,
        adapter_dim: int = 64,
        num_heads: int = 8,
        dropout: float = 0.1,
        pretrained: bool = True,
        num_pattern_tokens: int = 4,
        num_bottlenecks: int = 4,
    ):
        super().__init__()

        self.gnss_input_dim = gnss_input_dim
        self.traj_encoder = TrajectoryPatternEncoder(
            input_dim=gnss_input_dim,
            embed_dim=768,
            hidden_dim=384,
            num_pattern_tokens=num_pattern_tokens,
            dropout=dropout,
        )
        self.visual_encoder = VisualEncoder(adapter_dim=adapter_dim, pretrained=pretrained)
        self.fusion = TemporalAlignmentFusion(
            embed_dim=768,
            num_heads=num_heads,
            num_bottlenecks=num_bottlenecks,
            dropout=dropout,
        )

        self.prototype_bank = nn.Parameter(torch.randn(num_classes, 768) * 0.02)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, num_classes),
        )

        nn.init.xavier_uniform_(self.classifier[1].weight)
        nn.init.zeros_(self.classifier[1].bias)

    def _ensure_sequence_input(self, video: torch.Tensor, gnss: torch.Tensor) -> torch.Tensor:
        if gnss.ndim == 2:
            B, F = gnss.shape
            assert F == self.gnss_input_dim, (
                f"期望 gnss 最后一维 {self.gnss_input_dim}，实际: {F}"
            )
            T = video.shape[1]
            gnss = gnss.unsqueeze(1).expand(B, T, F)
        assert gnss.ndim == 3, f"期望 gnss 形状 (B, T, F) 或 (B, F)，实际: {gnss.shape}"
        return gnss

    def compute_aux_losses(self, fused_feature: torch.Tensor, labels: torch.Tensor = None):
        aux = self.fusion.get_aux_outputs()
        losses = self.fusion.get_aux_losses()

        proto_loss = fused_feature.new_tensor(0.0)
        if labels is not None:
            class_prototypes = self.prototype_bank[labels]
            proto_loss = 1.0 - nn.functional.cosine_similarity(
                fused_feature,
                class_prototypes,
                dim=-1,
            ).mean()

        return {
            'alignment_loss': losses['alignment_loss'],
            'balance_loss': losses['balance_loss'],
            'prototype_loss': proto_loss,
            'alpha_v': aux.get('alpha_v'),
            'alpha_t': aux.get('alpha_t'),
        }

    def forward_features(self, video: torch.Tensor, gnss: torch.Tensor):
        B, T = video.shape[:2]
        gnss = self._ensure_sequence_input(video, gnss)

        assert gnss.shape[:2] == (B, T), (
            f"轨迹窗口与视频帧数需一致，期望 {(B, T, self.gnss_input_dim)}，实际: {gnss.shape}"
        )

        traj_tokens, traj_global = self.traj_encoder(gnss)
        visual_tokens = self.visual_encoder(video)
        fused = self.fusion(traj_tokens, traj_global, visual_tokens)
        return fused

    def forward(self, video: torch.Tensor, gnss: torch.Tensor) -> torch.Tensor:
        fused = self.forward_features(video, gnss)
        logits = self.classifier(fused)
        return logits

    def forward_with_aux(self, video: torch.Tensor, gnss: torch.Tensor, labels: torch.Tensor = None):
        fused = self.forward_features(video, gnss)
        logits = self.classifier(fused)
        aux = self.compute_aux_losses(fused, labels)
        return logits, aux

    def summary(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        print("=" * 55)
        print("  TC-AdaptFormer / TAIF 参数统计")
        print("=" * 55)

        modules = {
            'TrajectoryEncoder': self.traj_encoder,
            'VisualEncoder': self.visual_encoder,
            'Fusion': self.fusion,
            'Classifier': self.classifier,
        }
        for name, mod in modules.items():
            m_total = sum(p.numel() for p in mod.parameters())
            m_train = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            print(f"  {name:<20} 总: {m_total/1e6:6.2f}M  可训练: {m_train/1e6:6.2f}M")

        print("-" * 55)
        print(f"  {'总参数':<20} {total/1e6:6.2f}M")
        print(f"  {'可训练参数':<20} {trainable/1e6:6.2f}M  ({trainable/total*100:.1f}%)")
        print(f"  {'冻结参数':<20} {frozen/1e6:6.2f}M  ({frozen/total*100:.1f}%)")
        print("=" * 55)
