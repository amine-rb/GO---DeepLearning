"""Architecture 7 – MobileNet (Depthwise Separable Convolutions).

Replaces standard convolutions with depthwise separable ones,
dramatically reducing parameters while retaining spatial resolution.

Parameter count: ≈ 23 k
"""

import torch
import torch.nn as nn
from models.base_model import GoModel


class DepthwiseSeparableConv(nn.Module):
    """Depthwise convolution followed by a pointwise (1×1) convolution."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride,
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.pw(self.dw(x))


class MobileNet(GoModel):
    """MobileNet-style network using depthwise separable convolutions."""

    def __init__(self, base_channels: int = 48, num_layers: int = 6):
        super().__init__()

        layers = [
            nn.Conv2d(self.NUM_PLANES, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        ]
        ch = base_channels
        for _ in range(num_layers):
            layers.append(DepthwiseSeparableConv(ch, ch))

        self.trunk = nn.Sequential(*layers)

        # Policy head: 1×1 conv → flatten to 361
        self.policy_head = nn.Conv2d(ch, 1, 1)

        # Value head: GAP → FC
        self.value_pool = nn.AdaptiveAvgPool2d(1)
        self.value_fc1 = nn.Linear(ch, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        h = self.trunk(x)

        policy = self.policy_head(h).flatten(1)

        v = self.value_pool(h).flatten(1)
        v = torch.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy, value
