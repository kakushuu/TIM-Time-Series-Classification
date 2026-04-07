import torch
import torch.nn as nn
import timm
from models.pet_modules import VanillaEncoder, AdaptFormer
from models.patch_encoder import PatchEncoder

TRAJ_DIM = 27   # 27 维特征（仿照 GAN-BiLSTM: 2 经纬度 + 5 运动 × 5 统计量）
TRAJ_SEQ = 512  # sliding-window length (frames) - 仿照 GAN-BiLSTM
BILSTM_HIDDEN = 384  # hidden per direction; 384*2 = 768 matches ViT dim


class AVmodel(nn.Module):
    """
    Multimodal Bottleneck Transformer for Trajectory + Video fusion.

    Modes:
      - 'multimodal':      Full MBT with cross-modal attention (trajectory + image)
      - 'trajectory_only': BiLSTM or PatchTST trajectory encoder → MLP classifier
      - 'image_only':      ViT image encoder → classifier (no trajectory)

    traj_arch:
      - 'bilstm':    Attention-BiLSTM encoder (default, Bug A1 fixed)
      - 'patchtst':  PatchTST-style patch Transformer encoder (B1)
    """
    def __init__(self, num_classes, num_latents, dim, mode='multimodal', traj_arch='bilstm',
                 bilstm_hidden=384, bilstm_layers=2):
        super(AVmodel, self).__init__()
        self.mode = mode
        self.traj_arch = traj_arch
        self.bilstm_hidden = bilstm_hidden
        self.bilstm_layers = bilstm_layers

        # ── Trajectory encoder (shared across modes) ──────────────────────────────
        if traj_arch == 'bilstm':
            # BiLSTM + attention pooling (Bug A1 fixed)
            self.traj_bilstm = nn.LSTM(
                input_size=TRAJ_DIM,
                hidden_size=bilstm_hidden,
                num_layers=bilstm_layers,
                bidirectional=True,
                batch_first=True,
                dropout=0.3,
            )
            # Attention pooling layer
            bilstm_output_dim = bilstm_hidden * 2  # BiLSTM bidirectional doubles hidden size
            self.traj_attn_w    = nn.Linear(bilstm_output_dim, 1)
            self.traj_cls_token = nn.Parameter(torch.zeros(1, 1, bilstm_output_dim))
            self.traj_layernorm = nn.LayerNorm(bilstm_output_dim)
        elif traj_arch == 'hierarchical_bilstm':
            # Hierarchical BiLSTM: short-term (all frames) + long-term (downsampled)
            # Short-term: captures fast motion patterns
            self.traj_bilstm_short = nn.LSTM(
                input_size=TRAJ_DIM,
                hidden_size=BILSTM_HIDDEN,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
                dropout=0.3,
            )
            # Long-term: captures global trajectory shape (every 10th frame)
            self.traj_bilstm_long = nn.LSTM(
                input_size=TRAJ_DIM,
                hidden_size=BILSTM_HIDDEN,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
                dropout=0.3,
            )
            # Attention pooling for both scales
            self.traj_attn_w_short = nn.Linear(768, 1)
            self.traj_attn_w_long = nn.Linear(768, 1)
            self.traj_layernorm_short = nn.LayerNorm(768)
            self.traj_layernorm_long = nn.LayerNorm(768)
            # Fusion: concat(768+768) → 768
            self.traj_fusion = nn.Linear(768 * 2, 768)
        elif traj_arch == 'traj_image':
            # Trajectory-as-Image: reshape (27, 512) → CNN input
            # Use ResNet-18 pretrained on ImageNet
            import torchvision.models as models
            self.traj_cnn = models.resnet18(pretrained=True)
            # Replace first conv: normally (3, 7, 7) for RGB, we need (1, 7, 7) for single channel
            # Or better: treat 27 features as 27 channels (like hyperspectral image)
            self.traj_cnn.conv1 = nn.Conv2d(TRAJ_DIM, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Replace final FC: 1000 → num_classes (will be done in _init_trajectory_only)
            # For now, remove the FC to get 512-dim features
            self.traj_cnn.fc = nn.Identity()  # Output: (bs, 512)
            # Freeze early layers
            for param in self.traj_cnn.parameters():
                param.requires_grad = False
            # Unfreeze last 2 blocks + conv1
            for param in self.traj_cnn.layer4.parameters():
                param.requires_grad = True
            for param in self.traj_cnn.conv1.parameters():
                param.requires_grad = True
        elif traj_arch == 'patchtst':
            # PatchTST: patch_size=16 → 32 tokens, 4-layer Transformer
            self.traj_patch_enc = PatchEncoder(
                seq_len=TRAJ_SEQ,
                n_features=TRAJ_DIM,
                patch_size=16,
                d_model=768,
                n_heads=8,
                n_layers=4,
                dropout=0.1,
            )
        else:
            raise ValueError(f"Unknown traj_arch: {traj_arch}. Choose from: bilstm, patchtst")

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
        """Trajectory-only: encoder → MLP classifier."""
        if self.traj_arch == 'traj_image':
            # CNN already outputs 512-dim features
            self.traj_encoder = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            self.classifier = nn.Linear(256, num_classes)
        else:
            # BiLSTM / hierarchical_bilstm / patchtst output bilstm_hidden*2
            self.traj_encoder = nn.Sequential(
                nn.Linear(self.bilstm_hidden * 2, 512),
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
        Trajectory encoder — supports bilstm, hierarchical_bilstm, traj_image, and patchtst.

        Args:
            x: (bs, T, 27)
        Returns:
            bilstm:              (bs, T+1, 768) — [cls_token, h_1..h_T] for MBT encoder
            hierarchical_bilstm: (bs, 1, 768)   — single fused token (short+long concat → fusion)
            traj_image:          (bs, 1, 512)   — CNN features (ResNet-18 output)
            patchtst:            (bs, 1, 768)   — single pooled token (wrapped for MBT compat)
        """
        B, T, _ = x.shape

        if self.traj_arch == 'bilstm':
            rnn_out, _ = self.traj_bilstm(x)              # (bs, T, bilstm_hidden*2)
            rnn_out = self.traj_layernorm(rnn_out)
            cls = self.traj_cls_token.expand(B, -1, -1)   # (bs, 1, bilstm_hidden*2)
            return torch.cat([cls, rnn_out], dim=1)        # (bs, T+1, bilstm_hidden*2)

        elif self.traj_arch == 'hierarchical_bilstm':
            # Short-term: all 512 frames
            short_out, _ = self.traj_bilstm_short(x)      # (bs, 512, 768)
            short_out = self.traj_layernorm_short(short_out)
            short_scores = torch.softmax(self.traj_attn_w_short(short_out), dim=1)  # (bs, 512, 1)
            short_pooled = (short_scores * short_out).sum(dim=1)  # (bs, 768)

            # Long-term: downsample to every 10th frame (512 → 51 frames)
            long_input = x[:, ::10, :]  # (bs, 51, 27)
            long_out, _ = self.traj_bilstm_long(long_input)  # (bs, 51, 768)
            long_out = self.traj_layernorm_long(long_out)
            long_scores = torch.softmax(self.traj_attn_w_long(long_out), dim=1)  # (bs, 51, 1)
            long_pooled = (long_scores * long_out).sum(dim=1)  # (bs, 768)

            # Fusion
            fused = torch.cat([short_pooled, long_pooled], dim=1)  # (bs, 1536)
            fused = self.traj_fusion(fused)  # (bs, 768)
            return fused.unsqueeze(1)  # (bs, 1, 768) — MBT compat

        elif self.traj_arch == 'traj_image':
            # Reshape (bs, T=512, 27) → (bs, 27, T=512) as "image"
            x = x.permute(0, 2, 1)  # (bs, 27, 512)
            # Add dummy spatial dimension: (bs, 27, 512) → (bs, 27, 512, 1)
            x = x.unsqueeze(-1)  # (bs, 27, 512, 1)
            # Interpolate to 224x224 (ResNet input size) - ensure on same device as model
            x = x.to(self.traj_cnn.conv1.weight.device)
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            # CNN forward pass
            features = self.traj_cnn(x)  # (bs, 512)
            return features.unsqueeze(1)  # (bs, 1, 512) — MBT compat

        else:  # patchtst
            out = self.traj_patch_enc(x)                   # (bs, 768)
            return out.unsqueeze(1)                        # (bs, 1, 768) — MBT compat

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

    def forward(self, x, y, return_cls_tokens=False):
        """
        x: (bs, T, 27)          trajectory sequence  (T = TRAJ_SEQ frames, default 512)
        y: (bs, F, 3, 224, 224) RGB frames

        Behavior depends on self.mode:
          - multimodal:      uses both x and y
          - trajectory_only: uses only x (y is ignored)
          - image_only:      uses only y (x is ignored)

        Args:
            return_cls_tokens: If True and mode='multimodal', returns (logits, traj_cls, rgb_cls)
                               Otherwise returns logits only
        """
        if self.mode == 'multimodal':
            x = self.forward_traj_features(x)
            y = self.forward_rgb_features(y)
            x, y = self.forward_encoder(x, y)
            logits = self.classifier((x + y) * 0.5)
            if return_cls_tokens:
                return logits, x, y  # x=traj_cls, y=rgb_cls (both 768-dim)
            return logits

        elif self.mode == 'trajectory_only':
            x = self.forward_traj_features(x)       # (bs, T+1, 768) or (bs, 1, 768) or (bs, 1, 512)
            if self.traj_arch == 'bilstm':
                # Attention pooling over BiLSTM hidden states (skip cls at pos 0)
                h = x[:, 1:]                                         # (bs, T, 768)
                scores = torch.softmax(self.traj_attn_w(h), dim=1)  # (bs, T, 1)
                x = (scores * h).sum(dim=1)                          # (bs, 768)
            else:
                # PatchTST / hierarchical_bilstm / traj_image already pooled — unwrap the single token
                x = x[:, 0]                                          # (bs, 768) or (bs, 512)
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
