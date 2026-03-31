"""Architecture 2 – SimpleCNN.

Three convolutional layers with batch normalisation followed by a policy
convolutional head and a value fully-connected head.

Parameter count: ≈ 80 k
"""

import torch
import torch.nn as nn
from models.base_model import GoModel


class SimpleCNN(GoModel):
    """Plain stack of 3×3 convolutions, no skip connections."""

    def __init__(self, channels: int = 64):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Conv2d(self.NUM_PLANES, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Policy head: 1×1 conv → flatten to 361
        self.policy_head = nn.Conv2d(channels, 1, 1)

        # Value head: GAP → FC
        self.value_pool = nn.AdaptiveAvgPool2d(1)
        self.value_fc1 = nn.Linear(channels, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        h = self.trunk(x)

        policy = self.policy_head(h).flatten(1)

        v = self.value_pool(h).flatten(1)
        v = torch.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy, value
