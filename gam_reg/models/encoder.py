from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
from torch import nn


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SharedRegistrationEncoder(nn.Module):
    """Shared-weight multi-scale 3D encoder returning [F0,F1,F2,F3]."""

    def __init__(self, in_channels: int = 1, channels: Sequence[int] = (16, 32, 64, 128)):
        super().__init__()
        if len(channels) != 4:
            raise ValueError("encoder must define four channel levels")
        self.channels = [int(c) for c in channels]
        self.level0 = ConvBlock3d(in_channels, self.channels[0])
        self.down1 = nn.Conv3d(self.channels[0], self.channels[1], kernel_size=3, stride=2, padding=1)
        self.level1 = ConvBlock3d(self.channels[1], self.channels[1])
        self.down2 = nn.Conv3d(self.channels[1], self.channels[2], kernel_size=3, stride=2, padding=1)
        self.level2 = ConvBlock3d(self.channels[2], self.channels[2])
        self.down3 = nn.Conv3d(self.channels[2], self.channels[3], kernel_size=3, stride=2, padding=1)
        self.level3 = ConvBlock3d(self.channels[3], self.channels[3])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.ndim != 5:
            raise AssertionError("encoder input must have shape [B,C,D,H,W]")
        f0 = self.level0(x)
        f1 = self.level1(self.down1(f0))
        f2 = self.level2(self.down2(f1))
        f3 = self.level3(self.down3(f2))
        return [f0, f1, f2, f3]
