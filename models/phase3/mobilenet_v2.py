"""Architecture 11 – MobileNetV2-style (Inverted Residuals with Linear Bottlenecks).

Uses inverted residual blocks: the narrow → wide → narrow bottleneck
with a linear (non-activated) output, plus depthwise separable convolutions.

Parameter count: ≈ 79 k
"""

import torch
import torch.nn as nn
from models.base_model import GoModel


class InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual block.

    Expands channels by `expand_ratio`, applies depthwise conv, then
    projects back with a linear (no activation) 1×1 conv.
    """

    def __init__(self, in_ch: int, out_ch: int, expand_ratio: int = 4):
        super().__init__()
        hidden_ch = in_ch * expand_ratio
        self.use_residual = (in_ch == out_ch)
        layers = []
        # Expand
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, hidden_ch, 1, bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU6(inplace=True),
            ]
        # Depthwise
        layers += [
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1,
                      groups=hidden_ch, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU6(inplace=True),
        ]
        # Project (linear)
        layers += [
            nn.Conv2d(hidden_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            return x + out
        return out


class MobileNetV2(GoModel):
    """MobileNetV2-style network for Go prediction."""

    def __init__(self, base_ch: int = 28, expand_ratio: int = 4):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(self.NUM_PLANES, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU6(inplace=True),
        )

        self.blocks = nn.Sequential(
            InvertedResidual(base_ch, base_ch, expand_ratio),
            InvertedResidual(base_ch, base_ch * 2, expand_ratio),
            InvertedResidual(base_ch * 2, base_ch * 2, expand_ratio),
            InvertedResidual(base_ch * 2, base_ch, expand_ratio),
            InvertedResidual(base_ch, base_ch, expand_ratio),
        )

        # Policy head: 1×1 conv → flatten to 361
        self.policy_head = nn.Conv2d(base_ch, 1, 1)

        # Value head: GAP → FC
        self.value_pool = nn.AdaptiveAvgPool2d(1)
        self.value_fc1 = nn.Linear(base_ch, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)

        policy = self.policy_head(h).flatten(1)

        v = self.value_pool(h).flatten(1)
        v = torch.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy, value
