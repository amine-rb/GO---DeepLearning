"""Evaluation metrics for Go prediction models."""

import torch
import numpy as np
from typing import Dict


def policy_accuracy(policy_logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 policy accuracy (fraction of moves predicted correctly)."""
    preds = policy_logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def policy_top_k_accuracy(policy_logits: torch.Tensor, targets: torch.Tensor,
                           k: int = 5) -> float:
    """Top-k policy accuracy."""
    _, top_k = policy_logits.topk(k, dim=1)
    targets_expanded = targets.unsqueeze(1).expand_as(top_k)
    correct = top_k.eq(targets_expanded).any(dim=1)
    return correct.float().mean().item()


def value_mae(value_pred: torch.Tensor, value_targets: torch.Tensor) -> float:
    """Mean absolute error on value predictions."""
    return (value_pred.squeeze(1) - value_targets).abs().mean().item()


def compute_metrics(policy_logits: torch.Tensor, value_pred: torch.Tensor,
                    policy_targets: torch.Tensor, value_targets: torch.Tensor,
                    top_k: int = 5) -> Dict[str, float]:
    """Compute all evaluation metrics and return a dictionary."""
    return {
        "policy_acc": policy_accuracy(policy_logits, policy_targets),
        f"policy_top{top_k}_acc": policy_top_k_accuracy(policy_logits, policy_targets, top_k),
        "value_mae": value_mae(value_pred, value_targets),
    }
