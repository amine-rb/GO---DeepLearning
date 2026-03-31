"""Architecture 6 – SENet (Squeeze-and-Excitation Network).

Adds channel-wise attention (SE blocks) on top of a residual backbone.

Parameter count: ≈ 98 k
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import GoModel


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, reduced)
        self.fc2 = nn.Linear(reduced, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.gap(x).view(b, c)
        s = torch.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.view(b, c, 1, 1)


class SEResBlock(nn.Module):
    """Residual block with SE attention."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, reduction)

    def forward(self, x):
        h = torch.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = self.se(h)
        return torch.relu(x + h)


class SENet(GoModel):
    """Stack of SE-ResBlocks for Go prediction."""

    def __init__(self, channels: int = 32, num_blocks: int = 5, reduction: int = 8):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(self.NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            *[SEResBlock(channels, reduction) for _ in range(num_blocks)]
        )

        # Policy head: 1×1 conv → flatten to 361
        self.policy_head = nn.Conv2d(channels, 1, 1)

        # Value head: GAP → FC
        self.value_pool = nn.AdaptiveAvgPool2d(1)
        self.value_fc1 = nn.Linear(channels, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)

        policy = self.policy_head(h).flatten(1)

        v = self.value_pool(h).flatten(1)
        v = torch.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy, value
