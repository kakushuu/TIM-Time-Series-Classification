#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BiLSTM with Attention for Trajectory Classification
Adapted for 11-class agricultural activity classification
"""

import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Attention mechanism for sequence aggregation"""
    def __init__(self, rnn_size: int):
        super(Attention, self).__init__()
        self.w = nn.Linear(rnn_size, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, H):
        """
        Args:
            H: (batch_size, seq_len, rnn_size)
        Returns:
            r: (batch_size, rnn_size)
            alpha: (batch_size, seq_len) attention weights
        """
        M = self.tanh(H)  # (batch_size, seq_len, rnn_size)

        alpha = self.w(M).squeeze(2)  # (batch_size, seq_len)
        alpha = self.softmax(alpha)  # (batch_size, seq_len)

        r = H * alpha.unsqueeze(2)  # (batch_size, seq_len, rnn_size)
        r = r.sum(dim=1)  # (batch_size, rnn_size)

        return r, alpha


class AttBiLSTM(nn.Module):
    """Attention-based Bidirectional LSTM for trajectory classification"""
    def __init__(
            self,
            n_classes: int,
            emb_size: int,
            rnn_size: int,
            rnn_layers: int,
            dropout: float
    ):
        super(AttBiLSTM, self).__init__()
        self.rnn_size = rnn_size
        self.n_classes = n_classes

        # Bidirectional LSTM
        self.BiLSTM = nn.LSTM(
            emb_size, rnn_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0
        )

        # Attention mechanism
        self.attention = Attention(rnn_size)

        # Activation
        self.tanh = nn.Tanh()

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(rnn_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes)
        )

        # Normalization
        self.batchnorm = nn.BatchNorm1d(emb_size)
        self.layernorm = nn.LayerNorm(rnn_size)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, emb_size) trajectory features
        Returns:
            scores: (batch_size, n_classes) class logits
        """
        batch_size, seq_len, _ = x.shape

        # Batch normalization
        x = x.transpose(1, 2)  # (batch_size, emb_size, seq_len)
        x = self.batchnorm(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, emb_size)

        # BiLSTM encoding
        rnn_out, _ = self.BiLSTM(x)  # (batch_size, seq_len, 2*rnn_size)

        # Combine forward and backward directions
        H = rnn_out[:, :, :self.rnn_size] + rnn_out[:, :, self.rnn_size:]
        H = self.layernorm(H)

        # Attention aggregation
        r, alphas = self.attention(H)  # (batch_size, rnn_size), (batch_size, seq_len)
        h = self.tanh(r)  # (batch_size, rnn_size)

        # Classification
        scores = self.classifier(h)

        return scores


class AttBiLSTM_Multimodal(nn.Module):
    """Multimodal version with trajectory + image features"""
    def __init__(
            self,
            n_classes: int,
            traj_emb_size: int,
            img_feat_size: int,
            rnn_size: int,
            rnn_layers: int,
            dropout: float
    ):
        super(AttBiLSTM_Multimodal, self).__init__()
        self.rnn_size = rnn_size
        self.n_classes = n_classes

        # Trajectory BiLSTM
        self.traj_bilstm = nn.LSTM(
            traj_emb_size, rnn_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0
        )

        # Image feature projection
        self.img_proj = nn.Sequential(
            nn.Linear(img_feat_size, rnn_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(rnn_size * 2, rnn_size * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Attention mechanism
        self.attention = Attention(rnn_size)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(rnn_size * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes)
        )

        # Normalization
        self.batchnorm = nn.BatchNorm1d(traj_emb_size)
        self.layernorm = nn.LayerNorm(rnn_size)

    def forward(self, traj, img_feat):
        """
        Args:
            traj: (batch_size, seq_len, traj_emb_size) trajectory features
            img_feat: (batch_size, img_feat_size) image features
        Returns:
            scores: (batch_size, n_classes) class logits
        """
        batch_size = traj.shape[0]

        # Trajectory processing
        traj = traj.transpose(1, 2)
        traj = self.batchnorm(traj)
        traj = traj.transpose(1, 2)

        traj_rnn_out, _ = self.traj_bilstm(traj)  # (batch_size, seq_len, 2*rnn_size)

        # Attention aggregation
        H = traj_rnn_out[:, :, :self.rnn_size] + traj_rnn_out[:, :, self.rnn_size:]
        H = self.layernorm(H)

        r, _ = self.attention(H)  # (batch_size, rnn_size)

        # Image feature projection
        img_feat_proj = self.img_proj(img_feat)  # (batch_size, rnn_size)

        # Fusion
        fused = torch.cat([r, img_feat_proj], dim=1)  # (batch_size, 2*rnn_size)
        fused = self.fusion(fused)

        # Classification
        scores = self.classifier(fused)

        return scores
