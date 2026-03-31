"""Architecture 10 – WideResNet.

A wider (more channels) but shallower ResNet, following the approach of
Zagoruyko & Komodakis (2016). Fewer layers but wider feature maps.

Parameter count: ≈ 73 k
"""

import torch
import torch.nn as nn
from models.base_model import GoModel


class WideResBlock(nn.Module):
    """Wide residual block with dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class WideResNet(GoModel):
    """Wide ResNet with widen_factor controlling channel multiplier."""

    def __init__(self, depth: int = 2, widen_factor: int = 2, dropout: float = 0.2):
        super().__init__()
        base_ch = 16
        widths = [base_ch, base_ch * widen_factor, base_ch * widen_factor]

        self.stem = nn.Conv2d(self.NUM_PLANES, widths[0], 3, padding=1, bias=False)

        group1_layers = [WideResBlock(widths[0], widths[1], dropout)]
        for _ in range(depth - 1):
            group1_layers.append(WideResBlock(widths[1], widths[1], dropout))
        self.group1 = nn.Sequential(*group1_layers)

        group2_layers = [WideResBlock(widths[1], widths[2], dropout)]
        for _ in range(depth - 1):
            group2_layers.append(WideResBlock(widths[2], widths[2], dropout))
        self.group2 = nn.Sequential(*group2_layers)

        self.bn_out = nn.BatchNorm2d(widths[2])
        final_ch = widths[2]

        # Policy head: 1×1 conv → flatten to 361
        self.policy_head = nn.Conv2d(final_ch, 1, 1)

        # Value head: GAP → FC
        self.value_pool = nn.AdaptiveAvgPool2d(1)
        self.value_fc1 = nn.Linear(final_ch, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        h = self.stem(x)
        h = self.group1(h)
        h = self.group2(h)
        h = torch.relu(self.bn_out(h))

        policy = self.policy_head(h).flatten(1)

        v = self.value_pool(h).flatten(1)
        v = torch.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy, value
