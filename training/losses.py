"""Loss functions for dual-head Go prediction models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualHeadLoss(nn.Module):
    """Combined policy (cross-entropy) + value (MSE) loss.

    Total loss = policy_weight * CE(policy) + value_weight * MSE(value)
    """

    def __init__(self, policy_weight: float = 1.0, value_weight: float = 1.0):
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight

    def forward(self, policy_logits: torch.Tensor, value_pred: torch.Tensor,
                policy_targets: torch.Tensor, value_targets: torch.Tensor):
        """Compute total loss.

        Args:
            policy_logits: (B, 361) raw logits
            value_pred:    (B, 1)  tanh-activated prediction
            policy_targets: (B,) integer move indices
            value_targets:  (B,) float in [-1, 1]

        Returns:
            total_loss, policy_loss, value_loss
        """
        policy_loss = F.cross_entropy(policy_logits, policy_targets)
        value_loss = F.mse_loss(value_pred.squeeze(1), value_targets)
        total = self.policy_weight * policy_loss + self.value_weight * value_loss
        return total, policy_loss, value_loss
