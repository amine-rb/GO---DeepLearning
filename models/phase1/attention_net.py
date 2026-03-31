"""Architecture 9 – AttentionNet (Non-local / Self-Attention).

Integrates a lightweight self-attention layer between convolutional blocks
to capture long-range spatial dependencies across the board.

Parameter count: ≈ 44 k
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import GoModel


class SpatialSelfAttention(nn.Module):
    """Lightweight spatial self-attention module.

    Projects the input to Q, K, V using 1×1 convolutions, computes scaled
    dot-product attention, and adds the result residually.
    """

    def __init__(self, channels: int, key_channels: int = 16):
        super().__init__()
        self.key_channels = key_channels
        self.query = nn.Conv2d(channels, key_channels, 1, bias=False)
        self.key = nn.Conv2d(channels, key_channels, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        Q = self.query(x).view(B, self.key_channels, N).permute(0, 2, 1)  # B×N×k
        K = self.key(x).view(B, self.key_channels, N)                       # B×k×N
        V = self.value(x).view(B, C, N).permute(0, 2, 1)                   # B×N×C

        attn = F.softmax(torch.bmm(Q, K) / (self.key_channels ** 0.5), dim=-1)  # B×N×N
        out = torch.bmm(attn, V).permute(0, 2, 1).view(B, C, H, W)
        return x + self.gamma * self.out_proj(out)


class AttentionNet(GoModel):
    """CNN backbone with a self-attention layer for Go prediction."""

    def __init__(self, channels: int = 32, num_conv_blocks: int = 3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(self.NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        conv_blocks = []
        for _ in range(num_conv_blocks):
            conv_blocks += [
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ]
        self.conv_blocks = nn.Sequential(*conv_blocks)

        self.attention = SpatialSelfAttention(channels, key_channels=16)

        self.post_attn = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
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
        h = self.stem(x)
        h = self.conv_blocks(h)
        h = self.attention(h)
        h = self.post_attn(h)

        policy = self.policy_head(h).flatten(1)

        v = self.value_pool(h).flatten(1)
        v = torch.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy, value
