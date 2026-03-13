import torch
import torch.nn as nn
import timm
from models.pet_modules import VanillaEncoder, AdaptFormer

TRAJ_DIM = 6   # 经度, 纬度, 间距(米), 深度, 速度, 方向角
TRAJ_SEQ = 8   # sliding-window length (frames)
BILSTM_HIDDEN = 384  # hidden per direction; 384*2 = 768 matches ViT dim


class AVmodel(nn.Module):
    """
    Multimodal Bottleneck Transformer for Trajectory + Video fusion.

    Modes:
      - 'multimodal':      Full MBT with cross-modal attention (trajectory + image)
      - 'trajectory_only': BiLSTM trajectory encoder → MLP classifier (no image)
      - 'image_only':      ViT image encoder → classifier (no trajectory)

    Trajectory branch uses an Attention-BiLSTM encoder that takes a sequence of
    TRAJ_SEQ frames (bs, T, 6) and produces a single 768-dim cls token, matching
    the ViT token dimension so the MBT cross-modal attention is unchanged.
    """
    def __init__(self, num_classes, num_latents, dim, mode='multimodal'):
        super(AVmodel, self).__init__()
        self.mode = mode

        # ── Common: BiLSTM trajectory encoder ────────────────────────────────────
        # Replaces the previous single-frame Linear projection.
        # Input:  (bs, T, 6)  →  Output: (bs, T+1, 768)  [cls + T hidden states]
        self.traj_bilstm = nn.LSTM(
            input_size=TRAJ_DIM,
            hidden_size=BILSTM_HIDDEN,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3,
        )
        self.traj_attn_w    = nn.Linear(768, 1)          # attention scoring
        self.traj_cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.traj_layernorm = nn.LayerNorm(768)

        # ── Mode-specific initialization ─────────────────────────────────────────
        if mode == 'multimodal':
            self._init_multimodal(num_classes, num_latents, dim)
        elif mode == 'trajectory_only':
            self._init_trajectory_only(num_classes)
        elif mode == 'image_only':
            self._init_image_only(num_classes)
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from: multimodal, trajectory_only, image_only")

    # ── Initialization methods ───────────────────────────────────────────────────

    def _init_multimodal(self, num_classes, num_latents, dim):
        """Full MBT architecture with cross-modal attention."""
        # v1: trajectory stream  (ViT blocks reused for MBT cross-attention)
        # v2: RGB visual stream
        self.v1 = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
        self.v2 = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)

        self.v1.pre_logits = nn.Identity()
        self.v2.pre_logits = nn.Identity()
        self.v1.head = nn.Identity()
        self.v2.head = nn.Identity()

        # Freeze ViT backbones
        self.v1.pos_embed.requires_grad = False
        for p in self.v1.patch_embed.proj.parameters(): p.requires_grad = False
        for p in self.v1.blocks.parameters():           p.requires_grad = False

        self.v2.pos_embed.requires_grad = False
        for p in self.v2.patch_embed.proj.parameters(): p.requires_grad = False
        for p in self.v2.blocks.parameters():           p.requires_grad = False

        # RGB conv projection, cls token, pos embed
        self.rgb_conv      = self.v2.patch_embed.proj
        self.rgb_pos_embed = self.v2.pos_embed
        self.rgb_cls_token = self.v2.cls_token

        # MBT encoder (12 blocks)
        encoder_layers = []
        for i in range(12):
            encoder_layers.append(
                AdaptFormer(num_latents=num_latents, dim=dim,
                            spec_enc=self.v1.blocks[i],
                            rgb_enc=self.v2.blocks[i])
            )
        self.audio_visual_blocks = nn.Sequential(*encoder_layers)

        # Final norm & classifier
        self.traj_post_norm = self.v1.norm
        self.rgb_post_norm  = self.v2.norm
        self.classifier     = nn.Linear(768, num_classes)

    def _init_trajectory_only(self, num_classes):
        """Trajectory-only: BiLSTM → attention pooling → MLP classifier."""
        self.traj_encoder = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(256, num_classes)

    def _init_image_only(self, num_classes):
        """Image-only: ViT backbone + classifier."""
        self.v2 = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
        self.v2.head = nn.Linear(768, num_classes)

        # Freeze backbone, only train classifier
        for p in self.v2.parameters():
            p.requires_grad = False
        for p in self.v2.head.parameters():
            p.requires_grad = True

    # ── Forward helpers ───────────────────────────────────────────────────

    def forward_traj_features(self, x):
        """
        BiLSTM trajectory encoder.

        Args:
            x: (bs, T, 6)  — sequence of T normalised trajectory frames
        Returns:
            (bs, T+1, 768) — [cls_token, h_1, ..., h_T]  compatible with MBT encoder
        """
        B, T, _ = x.shape

        # BiLSTM: (bs, T, 6) → (bs, T, 768)
        rnn_out, _ = self.traj_bilstm(x)          # (bs, T, 768)
        rnn_out = self.traj_layernorm(rnn_out)

        # Prepend learnable cls token
        cls = self.traj_cls_token.expand(B, -1, -1)   # (bs, 1, 768)
        x = torch.cat([cls, rnn_out], dim=1)           # (bs, T+1, 768)
        return x

    def forward_rgb_features(self, x):
        """x: (bs, F, 3, 224, 224) → (bs, 1+F*196, 768)"""
        B, no_of_frames, C, H, W = x.shape
        x = torch.reshape(x, (B * no_of_frames, C, H, W))
        x = self.rgb_conv(x)                                   # (bs*F, 768, 14, 14)

        _, dim, h, w = x.shape
        x = torch.reshape(x, (B, no_of_frames, dim, h, w))
        x = x.permute(0, 2, 1, 3, 4)                          # (bs, 768, F, 14, 14)
        x = torch.reshape(x, (B, dim, no_of_frames * h * w))  # (bs, 768, F*196)
        x = x.permute(0, 2, 1)                                 # (bs, F*196, 768)

        x = torch.cat([self.rgb_cls_token.expand(B, -1, -1), x], dim=1)   # (bs, 1+F*196, 768)
        x = x + nn.functional.interpolate(
            self.rgb_pos_embed.permute(0, 2, 1), x.shape[1], mode='linear'
        ).permute(0, 2, 1)
        return x

    def forward_encoder(self, x, y):
        for blk in self.audio_visual_blocks:
            x, y = blk(x, y)
        x = self.traj_post_norm(x)
        y = self.rgb_post_norm(y)
        x = x[:, 0]   # trajectory cls token
        y = y[:, 0]   # RGB cls token
        return x, y

    # ── Main forward ──────────────────────────────────────────────────────

    def forward(self, x, y):
        """
        x: (bs, T, 6)           trajectory sequence  (T = TRAJ_SEQ frames)
        y: (bs, F, 3, 224, 224) RGB frames

        Behavior depends on self.mode:
          - multimodal:      uses both x and y
          - trajectory_only: uses only x (y is ignored)
          - image_only:      uses only y (x is ignored)
        """
        if self.mode == 'multimodal':
            x = self.forward_traj_features(x)
            y = self.forward_rgb_features(y)
            x, y = self.forward_encoder(x, y)
            logits = self.classifier((x + y) * 0.5)
            return logits

        elif self.mode == 'trajectory_only':
            x = self.forward_traj_features(x)       # (bs, 2, 768)
            x = x[:, 0]                              # cls token (bs, 768)
            x = self.traj_encoder(x)                 # (bs, 256)
            logits = self.classifier(x)
            return logits

        elif self.mode == 'image_only':
            # y: (bs, F, 3, 224, 224) → (bs*F, 3, 224, 224)
            B, F, C, H, W = y.shape
            y = y.reshape(B * F, C, H, W)
            logits = self.v2(y)                      # (bs*F, num_classes)
            logits = logits.reshape(B, F, -1).mean(dim=1)  # average over frames
            return logits
