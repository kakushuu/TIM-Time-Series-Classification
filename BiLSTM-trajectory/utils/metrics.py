#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metrics for trajectory classification
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def scores(y_true, y_pred):
    """
    Compute classification metrics

    Args:
        y_true: true labels (n_samples,)
        y_pred: predicted labels (n_samples,)

    Returns:
        dict: metrics including accuracy, precision, recall, f1
    """
    # Compute basic metrics
    acc = accuracy_score(y_true, y_pred) * 100

    # Macro-averaged metrics
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100

    # Weighted-averaged metrics
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Per-class metrics
    n_classes = len(np.unique(y_true))
    per_class_metrics = {}
    for i in range(n_classes):
        per_class_metrics[f'class_{i}'] = {
            'precision': precision_score(y_true, y_pred, labels=[i], average='micro', zero_division=0) * 100,
            'recall': recall_score(y_true, y_pred, labels=[i], average='micro', zero_division=0) * 100,
            'f1_score': f1_score(y_true, y_pred, labels=[i], average='micro', zero_division=0) * 100
        }

    return {
        'accuracy': round(float(acc), 2),
        'precision_macro': round(float(precision_macro), 2),
        'recall_macro': round(float(recall_macro), 2),
        'f1_macro': round(float(f1_macro), 2),
        'precision_weighted': round(float(precision_weighted), 2),
        'recall_weighted': round(float(recall_weighted), 2),
        'f1_weighted': round(float(f1_weighted), 2),
        'per_class': per_class_metrics,
        'confusion_matrix': cm.tolist()
    }
