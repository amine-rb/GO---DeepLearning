"""Architecture 8 – DilatedCNN (Dilated / Atrous Convolutions).

Uses dilated convolutions to capture multi-scale context without
increasing parameters or reducing spatial resolution.

Parameter count: ≈ 74 k
"""

import torch
import torch.nn as nn
from models.base_model import GoModel


class DilatedBlock(nn.Module):
    """Residual block with dilated convolution."""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return torch.relu(x + self.block(x))


class DilatedCNN(GoModel):
    """Multi-scale dilated CNN for Go prediction."""

    def __init__(self, channels: int = 28):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(self.NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Blocks with increasing dilation: 1, 2, 4, 2, 1
        self.blocks = nn.Sequential(
            DilatedBlock(channels, dilation=1),
            DilatedBlock(channels, dilation=2),
            DilatedBlock(channels, dilation=4),
            DilatedBlock(channels, dilation=2),
            DilatedBlock(channels, dilation=1),
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
