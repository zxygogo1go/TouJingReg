from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def _as_shape3(spatial_shape: Sequence[int]) -> Tuple[int, int, int]:
    if len(spatial_shape) != 3:
        raise ValueError("spatial_shape must be a 3-tuple/list in D,H,W order")
    d, h, w = int(spatial_shape[0]), int(spatial_shape[1]), int(spatial_shape[2])
    if d <= 0 or h <= 0 or w <= 0:
        raise ValueError("spatial dimensions must be positive")
    return d, h, w


def identity_grid(
    spatial_shape: Sequence[int],
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Create an absolute normalized grid in xyz order with shape [B,D,H,W,3]."""
    d, h, w = _as_shape3(spatial_shape)
    if dtype is None:
        dtype = torch.float32
    z = torch.linspace(-1.0, 1.0, d, device=device, dtype=dtype)
    y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    grid = torch.stack((xx, yy, zz), dim=-1)
    return grid.unsqueeze(0).expand(int(batch_size), d, h, w, 3).contiguous()


def assert_grid(grid: torch.Tensor, volume: Optional[torch.Tensor] = None) -> None:
    if grid.ndim != 5 or grid.shape[-1] != 3:
        raise AssertionError("grid must have shape [B,D,H,W,3] with xyz last dimension")
    if not torch.is_floating_point(grid):
        raise AssertionError("grid must be floating point")
    if volume is not None:
        if volume.ndim != 5:
            raise AssertionError("volume must have shape [B,C,D,H,W]")
        if grid.shape[0] != volume.shape[0]:
            raise AssertionError("grid and volume batch size mismatch")


def spatial_transform(
    volume: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "border",
    align_corners: bool = True,
) -> torch.Tensor:
    """Sample volume at absolute normalized xyz grid locations.

    The returned tensor is defined on the grid spatial shape. For registration,
    warping moving to fixed frame must pass the inverse transform phi_f2m.
    """
    assert_grid(grid, volume)
    if volume.shape[0] != grid.shape[0]:
        raise AssertionError("volume and grid batch size mismatch")
    return F.grid_sample(
        volume,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


def grid_to_channels(grid: torch.Tensor) -> torch.Tensor:
    assert_grid(grid)
    return grid.permute(0, 4, 1, 2, 3).contiguous()


def channels_to_grid(channels: torch.Tensor) -> torch.Tensor:
    if channels.ndim != 5 or channels.shape[1] != 3:
        raise AssertionError("channels must have shape [B,3,D,H,W]")
    return channels.permute(0, 2, 3, 4, 1).contiguous()


def compose_transforms(phi_a: torch.Tensor, phi_b: torch.Tensor) -> torch.Tensor:
    """Compose absolute transforms as phi_a o phi_b.

    Semantics: first apply phi_b, then sample/apply phi_a at that location.
    """
    assert_grid(phi_a)
    assert_grid(phi_b)
    if phi_a.shape != phi_b.shape:
        raise AssertionError("composed transforms must have identical shape")
    sampled = spatial_transform(grid_to_channels(phi_a), phi_b)
    return channels_to_grid(sampled)


def resize_grid(phi: torch.Tensor, spatial_shape: Sequence[int]) -> torch.Tensor:
    """Resize an absolute normalized grid to a new D,H,W resolution."""
    assert_grid(phi)
    d, h, w = _as_shape3(spatial_shape)
    resized = F.interpolate(
        grid_to_channels(phi),
        size=(d, h, w),
        mode="trilinear",
        align_corners=True,
    )
    return channels_to_grid(resized)


def sample_grid_at_points(phi: torch.Tensor, points_xyz: torch.Tensor) -> torch.Tensor:
    """Sample a transform/grid at normalized xyz points.

    Args:
        phi: [B,D,H,W,3] absolute normalized transform.
        points_xyz: [B,N,3] normalized xyz points.
    Returns:
        [B,N,3] sampled transform values.
    """
    assert_grid(phi)
    if points_xyz.ndim != 3 or points_xyz.shape[-1] != 3:
        raise AssertionError("points_xyz must have shape [B,N,3]")
    if points_xyz.shape[0] != phi.shape[0]:
        raise AssertionError("points and grid batch size mismatch")
    grid = points_xyz[:, :, None, None, :]
    sampled = spatial_transform(grid_to_channels(phi), grid)
    return sampled[:, :, :, 0, 0].permute(0, 2, 1).contiguous()
