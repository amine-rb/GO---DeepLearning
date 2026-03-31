"""Base model class for all Go prediction networks.

Every architecture inherits from GoModel and implements forward() to return
a (policy_logits, value) pair:

    policy_logits: (batch, 361)  – raw scores before softmax
    value:         (batch, 1)    – tanh-activated win probability in [-1, 1]
"""

from typing import Tuple

import torch
import torch.nn as nn


class GoModel(nn.Module):
    """Abstract base class for Go dual-head prediction models.

    Subclasses must implement forward() returning (policy_logits, value).
    """

    BOARD_SIZE: int = 19
    NUM_MOVES: int = 19 * 19  # 361
    NUM_PLANES: int = 4       # input feature planes

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_budget_ok(self, budget: int = 100_000) -> bool:
        return self.count_parameters() <= budget
