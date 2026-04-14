"""
时间对齐与长尾鲁棒融合模块

包含三个核心机制：
- TAM: 轨迹 token 对视频帧 token 的软时间对齐
- Bottleneck fusion: 利用瓶颈 token 汇聚跨模态共享信息
- Reliability gating: 根据模态质量动态调整视频/轨迹贡献
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAlignmentFusion(nn.Module):
    """
    TAIF-Net 融合层

    输入:
      traj_tokens:   (B, Lt, D)
      traj_global:   (B, D)
      visual_tokens: (B, Tv, P, D)

    输出:
      fused: (B, D)
      aux:   dict
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_bottlenecks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_bottlenecks = num_bottlenecks

        self.frame_pool = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

        self.traj_to_video_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.bottleneck_tokens = nn.Parameter(torch.randn(1, num_bottlenecks, embed_dim) * 0.02)
        self.bottleneck_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.reliability_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2),
        )

        self.classifier_prep = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

        self._last_aux = {}

    def _pool_frames(self, visual_tokens: torch.Tensor):
        B, Tv, P, D = visual_tokens.shape
        flat = visual_tokens.reshape(B * Tv, P, D)
        scores = self.frame_pool(flat)
        weights = torch.softmax(scores, dim=1)
        pooled = (flat * weights).sum(dim=1)
        return pooled.reshape(B, Tv, D)

    def _alignment_regularizer(self, attn_weights: torch.Tensor) -> torch.Tensor:
        if attn_weights.shape[1] < 2:
            return attn_weights.new_tensor(0.0)
        diff = attn_weights[:, 1:, :] - attn_weights[:, :-1, :]
        return diff.pow(2).mean()

    def _balance_regularizer(self, alpha_v: torch.Tensor, alpha_t: torch.Tensor) -> torch.Tensor:
        return ((alpha_v - 0.5).pow(2) + (alpha_t - 0.5).pow(2)).mean()

    def forward(self, traj_tokens: torch.Tensor, traj_global: torch.Tensor, visual_tokens: torch.Tensor):
        assert traj_tokens.ndim == 3, f"期望 traj_tokens 形状 (B, Lt, D)，实际: {traj_tokens.shape}"
        assert traj_global.ndim == 2, f"期望 traj_global 形状 (B, D)，实际: {traj_global.shape}"
        assert visual_tokens.ndim == 4, (
            f"期望 visual_tokens 形状 (B, Tv, P, D)，实际: {visual_tokens.shape}"
        )

        B = traj_tokens.shape[0]
        frame_tokens = self._pool_frames(visual_tokens)

        aligned_visual, attn_weights = self.traj_to_video_attn(
            query=traj_tokens,
            key=frame_tokens,
            value=frame_tokens,
            need_weights=True,
            average_attn_weights=True,
        )
        aligned_global = aligned_visual.mean(dim=1)

        bottlenecks = self.bottleneck_tokens.expand(B, -1, -1)
        shared_tokens = torch.cat([traj_tokens, aligned_visual], dim=1)
        bottleneck_out, _ = self.bottleneck_attn(
            query=bottlenecks,
            key=shared_tokens,
            value=shared_tokens,
            need_weights=False,
        )
        shared_global = bottleneck_out.mean(dim=1)

        gate_logits = self.reliability_gate(torch.cat([traj_global, aligned_global], dim=-1))
        gate = torch.softmax(gate_logits, dim=-1)
        alpha_t = gate[:, :1]
        alpha_v = gate[:, 1:]

        fused = alpha_t * traj_global + alpha_v * aligned_global + shared_global
        fused = self.classifier_prep(fused)

        self._last_aux = {
            'alignment_weights': attn_weights,
            'alpha_t': alpha_t,
            'alpha_v': alpha_v,
            'alignment_loss': self._alignment_regularizer(attn_weights),
            'balance_loss': self._balance_regularizer(alpha_v, alpha_t),
            'aligned_visual_global': aligned_global,
            'shared_global': shared_global,
        }
        return fused

    def get_aux_losses(self):
        return {
            'alignment_loss': self._last_aux.get('alignment_loss', 0.0),
            'balance_loss': self._last_aux.get('balance_loss', 0.0),
        }

    def get_aux_outputs(self):
        return self._last_aux


CrossAttentionFusion = TemporalAlignmentFusion
