"""Architecture 1 – MLP (Multi-Layer Perceptron) baseline.

Fully-connected network operating on the flattened 4×19×19 = 1444 input.
Shared trunk → policy head (361 outputs) + value head (1 output).

Parameter count: ≈ 97 k  (well within 100 k budget).
"""

import torch
import torch.nn as nn
from models.base_model import GoModel


class MLP(GoModel):
    """Baseline MLP with a single hidden layer."""

    def __init__(self, hidden: int = 52):
        super().__init__()
        in_features = self.NUM_PLANES * self.BOARD_SIZE * self.BOARD_SIZE  # 1444

        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
        )

        self.policy_head = nn.Linear(hidden, self.NUM_MOVES)
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        h = self.trunk(x)
        policy = self.policy_head(h)
        value = self.value_head(h)
        return policy, value
