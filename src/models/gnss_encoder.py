"""
轨迹模式编码器

将窗口级 GNSS 轨迹序列编码为：
- 轨迹 token 序列：用于时间对齐与跨模态融合
- 轨迹全局表示：用于分类与模态可信度估计

输入:
  gnss_seq: (B, T, F)

输出:
  traj_tokens:  (B, T + P, D)
  traj_global:  (B, D)
"""

import torch
import torch.nn as nn


class TrajectoryPatternEncoder(nn.Module):
    """
    多尺度轨迹编码器

    设计目标：
    - 局部分支捕捉短时运动模式
    - 全局分支建模长时作业状态
    - operation-pattern tokens 提供农业作业先验
    """

    def __init__(
        self,
        input_dim: int = 7,
        embed_dim: int = 768,
        hidden_dim: int = 384,
        num_layers: int = 1,
        num_pattern_tokens: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_pattern_tokens = num_pattern_tokens

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.local_branch = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.global_branch = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )

        self.pattern_tokens = nn.Parameter(torch.randn(1, num_pattern_tokens, embed_dim) * 0.02)
        self.pattern_fuser = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.global_pool = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _pool_sequence(self, traj_tokens: torch.Tensor) -> torch.Tensor:
        scores = self.global_pool(traj_tokens)
        weights = torch.softmax(scores, dim=1)
        return (traj_tokens * weights).sum(dim=1)

    def forward(self, gnss_seq: torch.Tensor):
        """
        Args:
            gnss_seq: (B, T, F)
        Returns:
            traj_tokens: (B, T + P, D)
            traj_global: (B, D)
        """
        assert gnss_seq.ndim == 3, f"期望 gnss_seq 形状 (B, T, F)，实际: {gnss_seq.shape}"
        B, _, F = gnss_seq.shape
        assert F == self.input_dim, f"期望最后一维 {self.input_dim}，实际: {F}"

        projected = self.input_proj(gnss_seq)

        local_feat = self.local_branch(projected.transpose(1, 2)).transpose(1, 2)
        global_feat, _ = self.global_branch(projected)

        gate = self.gate(torch.cat([local_feat, global_feat], dim=-1))
        traj_tokens = gate * local_feat + (1.0 - gate) * global_feat
        traj_tokens = self.norm(traj_tokens)

        pattern_tokens = self.pattern_tokens.expand(B, -1, -1)
        pattern_context, _ = self.pattern_fuser(
            query=pattern_tokens,
            key=traj_tokens,
            value=traj_tokens,
        )
        pattern_tokens = self.norm(pattern_tokens + pattern_context)

        pooled_traj = self._pool_sequence(traj_tokens)
        pooled_pattern = pattern_tokens.mean(dim=1)
        traj_global = self.norm(0.5 * (pooled_traj + pooled_pattern))

        return torch.cat([traj_tokens, pattern_tokens], dim=1), traj_global


GNSSEncoder = TrajectoryPatternEncoder
