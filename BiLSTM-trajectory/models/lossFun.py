#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Loss Functions for Trajectory Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: weighting factor (default 0.25)
        gamma: focusing parameter (default 2)
        logits: if True, inputs are logits; if False, inputs are probabilities
    """

    def __init__(self, alpha=0.25, gamma=2, logits=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits

    def forward(self, pred, target):
        """
        Args:
            pred: (batch_size, n_classes) predictions
            target: (batch_size, n_classes) one-hot encoded targets

        Returns:
            loss: scalar
        """
        if self.logits:
            pred_probs = F.softmax(pred, dim=1)
        else:
            pred_probs = pred

        # Clip probabilities to avoid log(0)
        pred_probs = torch.clamp(pred_probs, min=1e-7, max=1 - 1e-7)

        # Compute cross-entropy
        ce_loss = -target * torch.log(pred_probs)

        # Compute focal loss
        focal_loss = self.alpha * (1 - pred_probs) ** self.gamma * ce_loss

        # Sum over classes and average over batch
        loss = focal_loss.sum(dim=1).mean()

        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross-Entropy Loss for class imbalance

    Args:
        weights: (n_classes,) class weights
    """

    def __init__(self, weights=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        if weights is not None:
            self.weights = torch.tensor(weights, dtype=torch.float32)
        else:
            self.weights = None

    def forward(self, pred, target):
        """
        Args:
            pred: (batch_size, n_classes) logits
            target: (batch_size,) class indices

        Returns:
            loss: scalar
        """
        if self.weights is not None:
            self.weights = self.weights.to(pred.device)
            loss = F.cross_entropy(pred, target, weight=self.weights)
        else:
            loss = F.cross_entropy(pred, target)

        return loss
