from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from gam_reg.models.spatial_transformer import compose_transforms, identity_grid


class DiffeomorphicIntegrator(nn.Module):
    """Scaling-and-squaring integration for stationary velocity fields."""

    def __init__(self, steps: int = 7):
        super().__init__()
        if steps < 0:
            raise ValueError("steps must be non-negative")
        self.steps = int(steps)

    def integrate_velocity(self, velocity: torch.Tensor) -> torch.Tensor:
        if velocity.ndim != 5 or velocity.shape[1] != 3:
            raise AssertionError("velocity must have shape [B,3,D,H,W] in xyz channel order")
        b, _, d, h, w = velocity.shape
        identity = identity_grid(
            (d, h, w),
            batch_size=b,
            device=velocity.device,
            dtype=velocity.dtype,
        )
        phi = identity + velocity.permute(0, 2, 3, 4, 1) / float(2 ** self.steps)
        for _ in range(self.steps):
            phi = compose_transforms(phi, phi)
        return phi

    def forward(self, velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        phi_fwd = self.integrate_velocity(velocity)
        phi_inv = self.integrate_velocity(-velocity)
        return phi_fwd, phi_inv


def constant_velocity(
    spatial_shape: Sequence[int],
    displacement_xyz: torch.Tensor,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Utility used by tests and demos."""
    d, h, w = int(spatial_shape[0]), int(spatial_shape[1]), int(spatial_shape[2])
    if dtype is None:
        dtype = displacement_xyz.dtype
    disp = displacement_xyz.to(device=device, dtype=dtype).view(1, 3, 1, 1, 1)
    return disp.expand(batch_size, 3, d, h, w).contiguous()
