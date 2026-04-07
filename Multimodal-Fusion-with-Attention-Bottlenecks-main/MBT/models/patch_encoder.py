"""
PatchTST-style patch encoder for GPS trajectory sequences.

Splits T=512 timesteps into non-overlapping patches of size P,
projects each patch to d_model=768, then applies a Transformer encoder.
Final representation is mean-pooled over all patch tokens.

Reference: Nie et al., "A Time Series is Worth 64 Words", ICLR 2023.
"""

import torch
import torch.nn as nn
import math


class PatchEncoder(nn.Module):
    """
    Patch-based Transformer encoder for multivariate time series.

    Args:
        seq_len:    Input sequence length T (default 512)
        n_features: Number of input features F (default 27)
        patch_size: Number of timesteps per patch (default 16 → 32 patches)
        d_model:    Transformer hidden dim (default 768, matches ViT)
        n_heads:    Number of attention heads (default 8)
        n_layers:   Number of Transformer encoder layers (default 4)
        dropout:    Dropout rate (default 0.1)
    """

    def __init__(
        self,
        seq_len: int = 512,
        n_features: int = 27,
        patch_size: int = 16,
        d_model: int = 768,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size  # 512 // 16 = 32

        # Project each flattened patch to d_model
        self.patch_proj = nn.Linear(n_features * patch_size, d_model)

        # Learnable positional embeddings for n_patches positions
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (bs, T, F) — normalised trajectory sequence

        Returns:
            (bs, d_model) — mean-pooled patch representation
        """
        B, T, F = x.shape

        # Truncate or pad to a multiple of patch_size
        T_use = (T // self.patch_size) * self.patch_size
        x = x[:, :T_use, :]                              # (B, T_use, F)

        # Reshape into patches: (B, n_patches, patch_size * F)
        n_patches = T_use // self.patch_size
        x = x.reshape(B, n_patches, self.patch_size * F)  # (B, n_patches, P*F)

        # Project + add positional embeddings
        x = self.patch_proj(x)                            # (B, n_patches, d_model)
        x = x + self.pos_embed[:, :n_patches, :]

        # Transformer
        x = self.transformer(x)                           # (B, n_patches, d_model)
        x = self.norm(x)

        # Mean pool over patches → single vector
        x = x.mean(dim=1)                                 # (B, d_model)
        return x
