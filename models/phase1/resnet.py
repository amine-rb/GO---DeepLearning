"""Architecture 3 – ResNet (Residual Network).

A compact ResNet with residual blocks.  Uses the same dual-head output
convention (policy_logits, value).

Parameter count: ≈ 96 k
"""

import torch
import torch.nn as nn
from models.base_model import GoModel


class ResidualBlock(nn.Module):
    """Standard pre-activation residual block (BN → ReLU → Conv × 2)."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNet(GoModel):
    """Compact ResNet with configurable number of residual blocks."""

    def __init__(self, channels: int = 32, num_blocks: int = 5):
        super().__init__()

        self.stem = nn.Conv2d(self.NUM_PLANES, channels, 3, padding=1, bias=False)
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.bn_out = nn.BatchNorm2d(channels)

        # Policy head: 1×1 conv → flatten to 361
        self.policy_head = nn.Conv2d(channels, 1, 1)

        # Value head: GAP → FC
        self.value_pool = nn.AdaptiveAvgPool2d(1)
        self.value_fc1 = nn.Linear(channels, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)
        h = torch.relu(self.bn_out(h))

        policy = self.policy_head(h).flatten(1)

        v = self.value_pool(h).flatten(1)
        v = torch.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy, value
