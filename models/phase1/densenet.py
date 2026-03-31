"""Architecture 4 – DenseNet (Densely Connected Network).

Each layer receives feature maps from all preceding layers.
Uses a bottleneck growth-rate design to stay within the 100 k budget.

Parameter count: ≈ 89 k
"""

import torch
import torch.nn as nn
from models.base_model import GoModel


class DenseLayer(nn.Module):
    """Single dense layer: BN → ReLU → 1×1 Conv (bottleneck) → BN → ReLU → 3×3 Conv."""

    def __init__(self, in_channels: int, growth_rate: int):
        super().__init__()
        bottleneck = 4 * growth_rate
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, bottleneck, 1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, growth_rate, 3, padding=1, bias=False),
        )

    def forward(self, x):
        out = self.block(x)
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels: int, num_layers: int, growth_rate: int):
        super().__init__()
        layers = []
        ch = in_channels
        for _ in range(num_layers):
            layers.append(DenseLayer(ch, growth_rate))
            ch += growth_rate
        self.layers = nn.Sequential(*layers)
        self.out_channels = ch

    def forward(self, x):
        return self.layers(x)


class DenseNet(GoModel):
    """Compact DenseNet for Go with two dense blocks."""

    def __init__(self, init_channels: int = 16, growth_rate: int = 12, num_layers: int = 5):
        super().__init__()

        self.stem = nn.Conv2d(self.NUM_PLANES, init_channels, 3, padding=1, bias=False)

        block1 = DenseBlock(init_channels, num_layers, growth_rate)
        ch1 = block1.out_channels
        self.block1 = block1

        # Transition: reduce channels by half
        trans_ch = ch1 // 2
        self.transition = nn.Sequential(
            nn.BatchNorm2d(ch1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch1, trans_ch, 1, bias=False),
        )

        block2 = DenseBlock(trans_ch, num_layers, growth_rate)
        self.block2 = block2
        final_ch = block2.out_channels

        self.bn_out = nn.BatchNorm2d(final_ch)

        # Policy head: 1×1 conv → flatten to 361
        self.policy_head = nn.Conv2d(final_ch, 1, 1)

        # Value head: GAP → FC
        self.value_pool = nn.AdaptiveAvgPool2d(1)
        self.value_fc1 = nn.Linear(final_ch, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        h = self.stem(x)
        h = self.block1(h)
        h = self.transition(h)
        h = self.block2(h)
        h = torch.relu(self.bn_out(h))

        policy = self.policy_head(h).flatten(1)

        v = self.value_pool(h).flatten(1)
        v = torch.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy, value
