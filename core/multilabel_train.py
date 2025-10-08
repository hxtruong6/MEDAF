"""
Clean Multi-Label Training Module
All duplicate functions have been moved to training_utils.py
This module now only contains legacy wrappers for backward compatibility
"""

from .training_utils import (
    calculate_multilabel_attention_diversity,
    calculate_multilabel_accuracy,
    train_multilabel_standard,
)


def multiLabelAttnDiv(cams_list, targets, eps=1e-6):
    return calculate_multilabel_attention_diversity(cams_list, targets, eps)


def multiLabelAccuracy(predictions, targets, threshold=0.5):
    return calculate_multilabel_accuracy(predictions, targets, threshold)


def train_multilabel(train_loader, model, criterion, optimizer, args, device=None):
    return train_multilabel_standard(
        train_loader, model, criterion, optimizer, args, device
    )
