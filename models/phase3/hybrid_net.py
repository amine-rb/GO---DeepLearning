"""Architecture 14 – HybridNet (CNN + Self-Attention).

Combines local convolutional feature extraction with a global self-attention
layer, exploiting both local patterns and long-range board dependencies.

This is the best-performing architecture achieving ~44.56% policy accuracy.

Parameter count: ≈ 73 k
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import GoModel


class ConvBlock(nn.Module):
    """Conv 3×3 → BN → GELU residual block."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.gelu(x + self.block(x))


class MultiHeadSelfAttention(nn.Module):
    """Efficient multi-head self-attention operating on spatial tokens."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        # Reshape to sequence
        tokens = x.permute(0, 2, 3, 1).reshape(B, N, C)
        residual = tokens

        tokens = self.norm(tokens)
        qkv = self.qkv(tokens).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each B × heads × N × head_dim

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = residual + out
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)


class HybridNet(GoModel):
    """CNN + Self-Attention hybrid for Go prediction.

    This is the best-performing Phase 3 architecture.
    """

    def __init__(self, channels: int = 32, num_conv_blocks: int = 3,
                 num_heads: int = 4):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(self.NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        # Local feature extraction
        self.conv_stage = nn.Sequential(
            *[ConvBlock(channels) for _ in range(num_conv_blocks)]
        )

        # Global self-attention
        self.attn = MultiHeadSelfAttention(channels, num_heads=num_heads)

        # Post-attention conv
        self.post_attn = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        # Policy head: 1×1 conv → flatten to 361
        self.policy_head = nn.Conv2d(channels, 1, 1)

        # Value head: GAP → FC
        self.value_pool = nn.AdaptiveAvgPool2d(1)
        self.value_norm = nn.LayerNorm(channels, eps=1e-6)
        self.value_fc1 = nn.Linear(channels, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        h = self.stem(x)
        h = self.conv_stage(h)
        h = self.attn(h)
        h = self.post_attn(h)

        policy = self.policy_head(h).flatten(1)

        v = self.value_pool(h).flatten(1)
        v = self.value_norm(v)
        v = torch.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy, value
