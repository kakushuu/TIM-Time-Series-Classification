"""
GNSS 编码器模块
将 7维 GNSS 轨迹特征编码为与视觉 token 同维度的 Query 向量

前向传播:
  输入: gnss (B, 7)
  输出: query_embed (B, 768)
"""

import torch
import torch.nn as nn


class GNSSEncoder(nn.Module):
    """
    GNSS 特征 MLP 编码器

    Architecture:
      Linear(7 → 128) → GELU → Linear(128 → 768) → LayerNorm

    参数量约 0.1M，全部可训练。
    """

    def __init__(self, input_dim: int = 7, hidden_dim: int = 128, output_dim: int = 768):
        """
        Args:
            input_dim:  GNSS 输入维度（默认 7）
            hidden_dim: 隐藏层维度（默认 128）
            output_dim: 输出维度，与 ViT embedding 维度对齐（默认 768）
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        # Xavier 初始化线性层
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, gnss: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gnss: (B, 7)  — 归一化后的 GNSS 特征向量
        Returns:
            query_embed: (B, 768)  — Cross-attention 使用的 Query 向量
        """
        assert gnss.ndim == 2 and gnss.shape[1] == 7, \
            f"期望 gnss 形状 (B, 7)，实际: {gnss.shape}"
        return self.encoder(gnss)
