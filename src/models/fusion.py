"""
Cross-Attention 融合模块

用 GNSS Query 向视觉 token 序列做交叉注意力，
提取与当前农机状态（速度/深度/方向）最相关的视觉特征。

前向传播:
  输入: query_gnss (B, 768), visual_tokens (B, T*196, 768)
  输出: fused_feature (B, 768)
"""

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    GNSS-conditioned 跨模态注意力融合

    公式:
      F = MultiheadAttn(Q=Q_gnss, K=X_visual, V=X_visual)

    Q_gnss 由 GNSSEncoder 输出，作为"查询"向量
    X_visual 为 T 帧展平后的 patch token 序列（长度 T*196）
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.0):
        """
        Args:
            embed_dim: 注意力维度（与 ViT embedding 维度一致，默认 768）
            num_heads: 注意力头数（默认 12，每头 64 维）
            dropout:   注意力 dropout（默认 0）
        """
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim({embed_dim}) 必须能被 num_heads({num_heads}) 整除"

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,   # 使用 (B, seq, dim) 格式
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query_gnss: torch.Tensor,
        visual_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_gnss:    (B, 768)         — GNSS 编码的查询向量
            visual_tokens: (B, T*196, 768)  — 展平的视觉 patch token 序列
        Returns:
            fused:         (B, 768)         — 融合后特征向量
        """
        B = query_gnss.shape[0]
        assert query_gnss.shape == (B, 768), \
            f"期望 query_gnss 形状 (B, 768)，实际: {query_gnss.shape}"
        assert visual_tokens.ndim == 3 and visual_tokens.shape[0] == B, \
            f"期望 visual_tokens 形状 (B, seq, 768)，实际: {visual_tokens.shape}"

        # 扩展 Query 为序列维: (B, 768) → (B, 1, 768)
        q = query_gnss.unsqueeze(1)

        # 交叉注意力: Q=(B,1,768), K=V=(B,T*196,768) → (B,1,768)
        attn_out, _ = self.attn(query=q, key=visual_tokens, value=visual_tokens)

        # squeeze 并归一化: (B, 768)
        fused = self.norm(attn_out.squeeze(1))
        return fused
