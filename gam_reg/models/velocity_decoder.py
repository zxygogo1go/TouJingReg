from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from gam_reg.models.encoder import ConvBlock3d


class ResidualVelocityDecoder(nn.Module):
    """U-Net residual stationary velocity decoder conditioned on Gaussian priors."""

    def __init__(
        self,
        encoder_channels: Sequence[int] = (16, 32, 64, 128),
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
    ):
        super().__init__()
        if len(encoder_channels) != 4 or len(decoder_channels) != 4:
            raise ValueError("encoder_channels and decoder_channels must have four levels")
        c0, c1, c2, c3 = [int(c) for c in encoder_channels]
        d3, d2, d1, d0 = [int(c) for c in decoder_channels]
        self.level3 = ConvBlock3d(3 * c3 + 4, d3)
        self.up3 = nn.Conv3d(d3, d2, kernel_size=1)
        self.level2 = ConvBlock3d(d2 + 3 * c2 + 4, d2)
        self.up2 = nn.Conv3d(d2, d1, kernel_size=1)
        self.level1 = ConvBlock3d(d1 + 3 * c1, d1)
        self.up1 = nn.Conv3d(d1, d0, kernel_size=1)
        self.level0 = ConvBlock3d(d0 + 3 * c0, d0)
        self.velocity_head = nn.Conv3d(d0, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)

    @staticmethod
    def _feature_triplet(fm: torch.Tensor, ff: torch.Tensor) -> torch.Tensor:
        if fm.shape != ff.shape:
            raise AssertionError("moving and fixed feature shapes must match")
        return torch.cat([fm, ff, (fm - ff).abs()], dim=1)

    def forward(
        self,
        moving_features: List[torch.Tensor],
        fixed_features: List[torch.Tensor],
        u3: torch.Tensor,
        c3: torch.Tensor,
        u2: torch.Tensor,
        c2: torch.Tensor,
    ) -> torch.Tensor:
        if len(moving_features) != 4 or len(fixed_features) != 4:
            raise AssertionError("decoder expects four feature levels")
        f0m, f1m, f2m, f3m = moving_features
        f0f, f1f, f2f, f3f = fixed_features
        if tuple(u3.shape[-3:]) != tuple(f3m.shape[-3:]) or tuple(c3.shape[-3:]) != tuple(f3m.shape[-3:]):
            raise AssertionError("coarse priors must match level 3 feature resolution")
        if tuple(u2.shape[-3:]) != tuple(f2m.shape[-3:]) or tuple(c2.shape[-3:]) != tuple(f2m.shape[-3:]):
            raise AssertionError("middle priors must match level 2 feature resolution")
        x3 = self.level3(torch.cat([self._feature_triplet(f3m, f3f), u3, c3], dim=1))
        x2 = F.interpolate(x3, size=f2m.shape[-3:], mode="trilinear", align_corners=True)
        x2 = self.level2(torch.cat([self.up3(x2), self._feature_triplet(f2m, f2f), u2, c2], dim=1))
        x1 = F.interpolate(x2, size=f1m.shape[-3:], mode="trilinear", align_corners=True)
        x1 = self.level1(torch.cat([self.up2(x1), self._feature_triplet(f1m, f1f)], dim=1))
        x0 = F.interpolate(x1, size=f0m.shape[-3:], mode="trilinear", align_corners=True)
        x0 = self.level0(torch.cat([self.up1(x0), self._feature_triplet(f0m, f0f)], dim=1))
        velocity = self.velocity_head(x0)
        if velocity.shape[1] != 3:
            raise AssertionError("velocity must have three xyz channels")
        return velocity
