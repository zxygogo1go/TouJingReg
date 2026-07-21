from __future__ import annotations

from typing import Tuple

import torch


def smoothness_loss(velocity: torch.Tensor) -> torch.Tensor:
    """First-order diffusion regularization on normalized xyz velocity."""
    if velocity.ndim != 5 or velocity.shape[1] != 3:
        raise AssertionError("velocity must have shape [B,3,D,H,W]")
    dz = velocity[:, :, 1:, :, :] - velocity[:, :, :-1, :, :]
    dy = velocity[:, :, :, 1:, :] - velocity[:, :, :, :-1, :]
    dx = velocity[:, :, :, :, 1:] - velocity[:, :, :, :, :-1]
    return (dx.square().mean() + dy.square().mean() + dz.square().mean()) / 3.0


def normalized_grid_to_voxel_grid(phi: torch.Tensor) -> torch.Tensor:
    """Convert absolute normalized xyz grid to voxel-coordinate xyz grid."""
    if phi.ndim != 5 or phi.shape[-1] != 3:
        raise AssertionError("phi must have shape [B,D,H,W,3]")
    _, d, h, w, _ = phi.shape
    scale = phi.new_tensor([(w - 1) / 2.0, (h - 1) / 2.0, (d - 1) / 2.0])
    return (phi + 1.0) * scale


def jacobian_determinant(phi: torch.Tensor) -> torch.Tensor:
    """Jacobian determinant of absolute transform in voxel coordinates.

    The output is computed on the interior lattice with shape [B,D-2,H-2,W-2].
    For an identity transform the determinant is exactly one up to float error.
    """
    if phi.ndim != 5 or phi.shape[-1] != 3:
        raise AssertionError("phi must have shape [B,D,H,W,3]")
    if min(phi.shape[1:4]) < 3:
        raise ValueError("spatial dimensions must be at least 3 for central differences")
    vox = normalized_grid_to_voxel_grid(phi)
    d_dx = (vox[:, 1:-1, 1:-1, 2:, :] - vox[:, 1:-1, 1:-1, :-2, :]) * 0.5
    d_dy = (vox[:, 1:-1, 2:, 1:-1, :] - vox[:, 1:-1, :-2, 1:-1, :]) * 0.5
    d_dz = (vox[:, 2:, 1:-1, 1:-1, :] - vox[:, :-2, 1:-1, 1:-1, :]) * 0.5
    jac = torch.stack((d_dx, d_dy, d_dz), dim=-1)
    return torch.linalg.det(jac.float()).to(phi.dtype)


def jacobian_folding_penalty(phi_fwd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    det = jacobian_determinant(phi_fwd)
    penalty = torch.relu(-det).square().mean()
    folding_ratio = (det <= 0).to(det.dtype).mean()
    return penalty, folding_ratio
