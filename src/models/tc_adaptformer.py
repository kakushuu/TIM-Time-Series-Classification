"""
TC-AdaptFormer 完整模型
Trajectory-Conditioned AdaptFormer for Agricultural Activity Recognition

架构概述:
  video (B,T,3,224,224) → VisualEncoder(冻结ViT+可训Adapter) → (B,T*196,768)
  gnss  (B,7)           → GNSSEncoder(MLP)                   → (B,768)
                                                ↓ Cross-Attention
                                          fused (B,768)
                                                ↓ Classifier
                                         logits (B,11)

可训练参数约 3.65M（占总参数 4.1%），无3D卷积。
"""

import torch
import torch.nn as nn

from .gnss_encoder import GNSSEncoder
from .visual_encoder import VisualEncoder
from .fusion import CrossAttentionFusion


class TCAdaptFormer(nn.Module):
    """
    TC-AdaptFormer: 基于1Hz对齐轨迹-视觉融合的农机作业识别模型

    Input:
        video: (B, T, 3, 224, 224)  — T帧 RGB 图像序列（T=5）
        gnss:  (B, 7)               — 当前时刻 GNSS 特征（7维，已归一化）

    Output:
        logits: (B, num_classes)    — 各作业类别得分
    """

    def __init__(
        self,
        num_classes: int = 11,
        adapter_dim: int = 64,
        num_heads: int = 12,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        """
        Args:
            num_classes: 分类数（默认11类农业活动）
            adapter_dim: AdaptFormer 适配器瓶颈维度（默认64）
            num_heads:   Cross-attention 头数（默认12）
            dropout:     分类头 Dropout（默认0.1）
            pretrained:  是否加载 ViT-B16 预训练权重
        """
        super().__init__()

        # ── 子模块 ────────────────────────────────────────────────
        self.gnss_encoder    = GNSSEncoder(input_dim=7, hidden_dim=128, output_dim=768)
        self.visual_encoder  = VisualEncoder(adapter_dim=adapter_dim, pretrained=pretrained)
        self.fusion          = CrossAttentionFusion(embed_dim=768, num_heads=num_heads)
        self.classifier      = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, num_classes),
        )

        # 分类头初始化
        nn.init.xavier_uniform_(self.classifier[1].weight)
        nn.init.zeros_(self.classifier[1].bias)

    def forward(self, video: torch.Tensor, gnss: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            video: (B, T, 3, 224, 224)
            gnss:  (B, 7)
        Returns:
            logits: (B, num_classes)
        """
        B, T = video.shape[:2]
        assert gnss.shape == (B, 7), \
            f"期望 gnss 形状 (B, 7)，实际: {gnss.shape}"
        assert video.shape[1:] == (T, 3, 224, 224), \
            f"期望 video 形状 (B, T, 3, 224, 224)，实际: {video.shape}"

        # Step 1: GNSS → Query 向量 (B, 768)
        gnss_query = self.gnss_encoder(gnss)

        # Step 2: 视频帧 → patch token 序列 (B, T*196, 768)
        visual_tokens = self.visual_encoder(video)

        # Step 3: Cross-attention 融合 (B, 768)
        fused = self.fusion(gnss_query, visual_tokens)

        # Step 4: 分类 (B, num_classes)
        logits = self.classifier(fused)
        return logits

    def summary(self) -> None:
        """打印模型参数量统计"""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = total - trainable

        print("=" * 55)
        print("  TC-AdaptFormer 参数统计")
        print("=" * 55)

        modules = {
            'GNSSEncoder':         self.gnss_encoder,
            'VisualEncoder':       self.visual_encoder,
            'CrossAttentionFusion':self.fusion,
            'Classifier':          self.classifier,
        }
        for name, mod in modules.items():
            m_total = sum(p.numel() for p in mod.parameters())
            m_train = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            print(f"  {name:<25} 总: {m_total/1e6:6.2f}M  可训练: {m_train/1e6:6.2f}M")

        print("-" * 55)
        print(f"  {'总参数':<25} {total/1e6:6.2f}M")
        print(f"  {'可训练参数':<25} {trainable/1e6:6.2f}M  ({trainable/total*100:.1f}%)")
        print(f"  {'冻结参数':<25} {frozen/1e6:6.2f}M  ({frozen/total*100:.1f}%)")
        print("=" * 55)

        # 显存估算（FP32，前向）
        mem_mb = total * 4 / 1024 / 1024
        print(f"  参数显存估算 (FP32): {mem_mb:.0f} MB")
        print("=" * 55)
