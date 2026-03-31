"""Architecture 12 – ConvNeXt-style.

Modernised pure-CNN architecture inspired by Liu et al. (2022).
Key ideas: large depthwise kernels (7×7), LayerNorm instead of BatchNorm,
inverted bottleneck, GELU activations.

Parameter count: ≈ 65 k
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import GoModel


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block: DW 7×7 → LN → PW expand (×4) → GELU → PW project."""

    def __init__(self, dim: int, expand: int = 4):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=True)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, dim * expand)
        self.pw2 = nn.Linear(dim * expand, dim)
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-6)

    def forward(self, x):
        residual = x
        h = self.dw_conv(x)
        # (B, C, H, W) → (B, H, W, C) for LayerNorm and Linear
        h = h.permute(0, 2, 3, 1)
        h = self.norm(h)
        h = self.pw1(h)
        h = F.gelu(h)
        h = self.pw2(h)
        h = h * self.gamma
        h = h.permute(0, 3, 1, 2)
        return residual + h


class ConvNeXt(GoModel):
    """ConvNeXt-style network for Go prediction."""

    def __init__(self, dim: int = 40, num_blocks: int = 4):
        super().__init__()

        self.stem = nn.Conv2d(self.NUM_PLANES, dim, 3, padding=1)
        self.stem_norm = nn.LayerNorm(dim, eps=1e-6)

        self.blocks = nn.Sequential(*[ConvNeXtBlock(dim) for _ in range(num_blocks)])

        # Policy head: 1×1 conv → flatten to 361
        self.policy_norm = nn.LayerNorm(dim, eps=1e-6)
        self.policy_head = nn.Conv2d(dim, 1, 1)

        # Value head: GAP → FC
        self.value_norm = nn.LayerNorm(dim, eps=1e-6)
        self.value_pool = nn.AdaptiveAvgPool2d(1)
        self.value_fc1 = nn.Linear(dim, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def _apply_ln_spatial(self, x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        """Apply LayerNorm to a (B, C, H, W) tensor along the channel dim."""
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, C)
        x = norm(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x

    def forward(self, x):
        h = self.stem(x)
        h = self._apply_ln_spatial(h, self.stem_norm)
        h = self.blocks(h)

        # Policy head
        p = self._apply_ln_spatial(h, self.policy_norm)
        policy = self.policy_head(p).flatten(1)

        # Value head
        v = self._apply_ln_spatial(h, self.value_norm)
        v = self.value_pool(v).flatten(1)
        v = torch.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy, value
