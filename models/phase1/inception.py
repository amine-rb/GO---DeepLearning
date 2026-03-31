"""Architecture 5 – InceptionNet.

Applies parallel convolutions with different kernel sizes (1×1, 3×3, 5×5)
and max-pooling branches, concatenating the results (Inception module).

Parameter count: ≈ 15 k
"""

import torch
import torch.nn as nn
from models.base_model import GoModel


class InceptionModule(nn.Module):
    """Single Inception module with four branches."""

    def __init__(self, in_channels: int, out_1x1: int, out_3x3: int,
                 out_5x5: int, out_pool: int):
        super().__init__()
        # Branch 1: 1×1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, 1),
            nn.BatchNorm2d(out_1x1),
            nn.ReLU(inplace=True),
        )
        # Branch 2: 1×1 → 3×3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_3x3 // 2, 1),
            nn.BatchNorm2d(out_3x3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_3x3 // 2, out_3x3, 3, padding=1),
            nn.BatchNorm2d(out_3x3),
            nn.ReLU(inplace=True),
        )
        # Branch 3: 1×1 → 5×5  (implemented as two 3×3 for efficiency)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_5x5 // 2, 1),
            nn.BatchNorm2d(out_5x5 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_5x5 // 2, out_5x5, 3, padding=1),
            nn.BatchNorm2d(out_5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_5x5, out_5x5, 3, padding=1),
            nn.BatchNorm2d(out_5x5),
            nn.ReLU(inplace=True),
        )
        # Branch 4: 3×3 max-pool → 1×1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, 1),
            nn.BatchNorm2d(out_pool),
            nn.ReLU(inplace=True),
        )

    @property
    def out_channels(self):
        b1 = self.branch1[0].out_channels
        b2 = self.branch2[-3].out_channels
        b3 = self.branch3[-3].out_channels
        b4 = self.branch4[-3].out_channels
        return b1 + b2 + b3 + b4

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x),
                          self.branch3(x), self.branch4(x)], dim=1)


class InceptionNet(GoModel):
    """Two Inception modules stacked for Go prediction."""

    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(self.NUM_PLANES, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )

        self.inception1 = InceptionModule(24, out_1x1=16, out_3x3=24,
                                          out_5x5=8, out_pool=8)
        ch1 = 16 + 24 + 8 + 8  # 56

        self.inception2 = InceptionModule(ch1, out_1x1=16, out_3x3=24,
                                          out_5x5=8, out_pool=8)
        final_ch = 16 + 24 + 8 + 8  # 56

        # Policy head: 1×1 conv → flatten to 361
        self.policy_head = nn.Conv2d(final_ch, 1, 1)

        # Value head: GAP → FC
        self.value_pool = nn.AdaptiveAvgPool2d(1)
        self.value_fc1 = nn.Linear(final_ch, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        h = self.stem(x)
        h = self.inception1(h)
        h = self.inception2(h)

        policy = self.policy_head(h).flatten(1)

        v = self.value_pool(h).flatten(1)
        v = torch.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy, value
