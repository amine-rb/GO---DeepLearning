"""Architecture 13 – EfficientFormer-style.

Alternates between MetaFormer-style token mixing (pooling-based, lightweight)
and efficient feed-forward blocks, inspired by Li et al. (2022).

Parameter count: ≈ 84 k
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import GoModel


class PoolMixer(nn.Module):
    """Token mixing via average pooling (PoolFormer approach)."""

    def __init__(self, pool_size: int = 3):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2,
                                  count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class MetaFormerBlock(nn.Module):
    """MetaFormer block: token_mixing + channel MLP with LayerNorm."""

    def __init__(self, dim: int, expand: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.mixer = PoolMixer()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expand),
            nn.GELU(),
            nn.Linear(dim * expand, dim),
        )
        self.gamma1 = nn.Parameter(torch.ones(dim) * 1e-5)
        self.gamma2 = nn.Parameter(torch.ones(dim) * 1e-5)

    def _to_channel_last(self, x):
        return x.permute(0, 2, 3, 1)

    def _to_channel_first(self, x):
        return x.permute(0, 3, 1, 2)

    def forward(self, x):
        # Token mixing
        res = x
        h = self._to_channel_last(x)
        h = self.norm1(h)
        h = self._to_channel_first(h)
        h = self.mixer(h)
        h = self._to_channel_last(h)
        h = h * self.gamma1
        h = self._to_channel_first(h)
        x = res + h

        # Channel MLP
        res = x
        h = self._to_channel_last(x)
        h = self.norm2(h)
        h = self.mlp(h)
        h = h * self.gamma2
        h = self._to_channel_first(h)
        x = res + h

        return x


class EfficientFormer(GoModel):
    """EfficientFormer-style architecture for Go prediction."""

    def __init__(self, dim: int = 40, num_blocks: int = 6):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(self.NUM_PLANES, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(*[MetaFormerBlock(dim) for _ in range(num_blocks)])

        # Policy head: 1×1 conv → flatten to 361
        self.policy_norm = nn.LayerNorm(dim, eps=1e-6)
        self.policy_head = nn.Conv2d(dim, 1, 1)

        # Value head: GAP → FC
        self.value_norm = nn.LayerNorm(dim, eps=1e-6)
        self.value_pool = nn.AdaptiveAvgPool2d(1)
        self.value_fc1 = nn.Linear(dim, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def _apply_ln_spatial(self, x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, C)
        x = norm(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)

        # Policy
        p = self._apply_ln_spatial(h, self.policy_norm)
        policy = self.policy_head(p).flatten(1)

        # Value
        v = self._apply_ln_spatial(h, self.value_norm)
        v = self.value_pool(v).flatten(1)
        v = torch.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy, value
