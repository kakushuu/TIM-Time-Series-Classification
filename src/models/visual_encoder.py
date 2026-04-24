"""
视觉编码器模块（含 AdaptFormer 适配器）

使用冻结的 ViT-B16 对 T 帧图像独立编码，
并在每个 transformer block 的 FFN 后插入可训练的 AdaptFormer 适配器。

前向传播:
  输入:  video (B, T, 3, 224, 224)
  输出:  tokens (B, T, 196, 768)   — 保留时间结构的 patch token 序列
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import timm

# 将项目根目录加入 sys.path，以便复用 MBT 现有适配器代码
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MBT_PATH = _PROJECT_ROOT / 'Multimodal-Fusion-with-Attention-Bottlenecks-main' / 'MBT'


class SingleStreamAdapter(nn.Module):
    """
    单流 AdaptFormer 适配器
    结构: down(768→dim) → QuickGELU → Dropout → up(dim→768)
    插入 ViT FFN 之后: output = ffn(x) + scale * adapter(x)

    这是对原始 pet_modules.py 中 AdaptFormer 的单流简化版本。
    """

    def __init__(self, embed_dim: int = 768, adapter_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.down = nn.Linear(embed_dim, adapter_dim)
        self.up   = nn.Linear(adapter_dim, embed_dim)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(1))

        # 初始化：down 用 Xavier，up 用零（保证初始时适配器输出为零）
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., 768)  →  adapter_out: (..., 768)"""
        return self.up(self.drop(self.act(self.down(x)))) * self.scale


class ViTBlockWithAdapter(nn.Module):
    """
    ViT transformer block + AdaptFormer 适配器包装器

    将原始 ViT block 的 FFN 输出与 Adapter 输出相加：
      x = x + attn(norm1(x))
      x = x + mlp(norm2(x)) + adapter(norm2(x))
    """

    def __init__(self, vit_block: nn.Module, adapter_dim: int = 64):
        super().__init__()
        self.norm1  = vit_block.norm1
        self.attn   = vit_block.attn
        self.norm2  = vit_block.norm2
        self.mlp    = vit_block.mlp
        # ViT-B16 可能有 ls1/ls2（layer scale），需一起保留
        self.ls1    = getattr(vit_block, 'ls1', None)
        self.ls2    = getattr(vit_block, 'ls2', None)
        self.drop_path = getattr(vit_block, 'drop_path', nn.Identity())
        self.adapter = SingleStreamAdapter(embed_dim=768, adapter_dim=adapter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention 分支
        attn_out = self.attn(self.norm1(x))
        if self.ls1 is not None:
            attn_out = self.ls1(attn_out)
        x = x + self.drop_path(attn_out)

        # FFN + Adapter 分支
        norm2_x = self.norm2(x)
        mlp_out = self.mlp(norm2_x)
        if self.ls2 is not None:
            mlp_out = self.ls2(mlp_out)
        x = x + self.drop_path(mlp_out) + self.adapter(norm2_x)
        return x


class VisualEncoder(nn.Module):
    """
    冻结 ViT-B16 + AdaptFormer 视觉编码器

    对 T 帧图像独立编码后展平输出 patch token 序列。

    参数冻结策略:
      - ViT-B16 所有原始参数: requires_grad = False
      - AdaptFormer adapter 参数: requires_grad = True
    """

    def __init__(self, adapter_dim: int = 64, pretrained: bool = True, pretrained_path: str = ""):
        """
        Args:
            adapter_dim: AdaptFormer 适配器瓶颈维度（默认 64）
            pretrained:  是否加载 ImageNet 预训练权重
            pretrained_path: 本地 ViT-B/16 权重路径；提供后不联网下载
        """
        super().__init__()

        # 加载 ViT-B16
        local_pretrained = bool(pretrained_path)
        vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained and not local_pretrained)
        if local_pretrained:
            self._load_local_pretrained(vit, Path(pretrained_path))
        vit.head = nn.Identity()      # 移除分类头
        vit.pre_logits = nn.Identity() if hasattr(vit, 'pre_logits') else nn.Identity()

        # 冻结所有 ViT 参数
        for p in vit.parameters():
            p.requires_grad = False

        # Patch embedding 和位置编码（冻结）
        self.patch_embed = vit.patch_embed
        self.cls_token   = vit.cls_token
        self.pos_embed   = vit.pos_embed
        self.pos_drop    = vit.pos_drop
        self.norm        = vit.norm

        # 将每个 ViT block 替换为带 Adapter 的版本
        # Adapter 参数可训练
        self.blocks = nn.ModuleList([
            ViTBlockWithAdapter(block, adapter_dim=adapter_dim)
            for block in vit.blocks
        ])

        # 打印参数量统计
        total   = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[VisualEncoder] 总参数: {total/1e6:.2f}M | 可训练: {trainable/1e6:.2f}M")

    def _load_local_pretrained(self, vit: nn.Module, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"local ViT pretrained weights not found: {path}")
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state = load_file(str(path))
        else:
            state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            for key in ("model", "state_dict", "model_state", "model_state_dict"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        if not isinstance(state, dict):
            raise TypeError(f"unsupported pretrained checkpoint format: {path}")
        cleaned = {}
        for key, value in state.items():
            new_key = key
            for prefix in ("module.", "model.", "visual."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            cleaned[new_key] = value
        incompatible = vit.load_state_dict(cleaned, strict=False)
        print(
            f"[VisualEncoder] Loaded local pretrained weights: {path} "
            f"(missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)})"
        )

    def encode_single_frame(self, x: torch.Tensor) -> torch.Tensor:
        """
        对单帧编码

        Args:
            x: (B, 3, 224, 224)
        Returns:
            patch_tokens: (B, 196, 768)  — 去掉 CLS token
        """
        B = x.shape[0]
        # Patch embedding: (B, 3, 224, 224) → (B, 196, 768)
        x = self.patch_embed(x)
        # 拼接 CLS token: (B, 197, 768)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        # 加位置编码
        x = self.pos_drop(x + self.pos_embed)
        # 12 个 transformer block
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # 返回 patch tokens（丢弃 CLS）
        return x[:, 1:, :]  # (B, 196, 768)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: (B, T, 3, 224, 224)
        Returns:
            tokens: (B, T, 196, 768)  — 保留时间维的多帧 patch token 序列
        """
        B, T, C, H, W = video.shape
        assert H == W == 224, f"期望 224×224 输入，实际: {H}×{W}"

        # reshape 为 (B*T, 3, 224, 224) 并行处理
        frames = video.view(B * T, C, H, W)
        tokens = self.encode_single_frame(frames)   # (B*T, 196, 768)
        tokens = tokens.reshape(B, T, 196, 768)     # (B, T, 196, 768)
        return tokens
