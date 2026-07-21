from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch


def _spacing_dhw_tensor(
    spacing_dhw: Optional[torch.Tensor | Sequence[float]],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if spacing_dhw is None:
        spacing = torch.ones((batch_size, 3), device=device, dtype=torch.float32)
    else:
        spacing = torch.as_tensor(spacing_dhw, device=device, dtype=torch.float32)
        if spacing.ndim == 1:
            if spacing.numel() != 3:
                raise ValueError("spacing_dhw must contain three values")
            spacing = spacing.view(1, 3).expand(batch_size, 3)
        elif spacing.ndim == 2 and spacing.shape == (batch_size, 3):
            pass
        else:
            raise ValueError("spacing_dhw must have shape [3] or [B,3]")
    if not bool(torch.isfinite(spacing).all()) or bool((spacing <= 0).any()):
        raise ValueError("spacing_dhw values must be finite and positive")
    return spacing


def normalized_velocity_to_physical(
    velocity: torch.Tensor,
    spacing_dhw: Optional[torch.Tensor | Sequence[float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert normalized xyz velocity to millimetres and return DHW spacing."""
    if velocity.ndim != 5 or velocity.shape[1] != 3:
        raise AssertionError("velocity must have shape [B,3,D,H,W]")
    with torch.amp.autocast(device_type=velocity.device.type, enabled=False):
        velocity_f = velocity.float()
        b, _, d, h, w = velocity_f.shape
        spacing = _spacing_dhw_tensor(spacing_dhw, b, velocity_f.device)
        spacing_xyz = spacing.flip(dims=(1,))
        voxel_scale_xyz = velocity_f.new_tensor(
            [(w - 1) / 2.0, (h - 1) / 2.0, (d - 1) / 2.0]
        ).view(1, 3, 1, 1, 1)
        physical_velocity = velocity_f * voxel_scale_xyz * spacing_xyz.view(b, 3, 1, 1, 1)
        return physical_velocity, spacing


def smoothness_loss(
    velocity: torch.Tensor,
    spacing_dhw: Optional[torch.Tensor | Sequence[float]] = None,
) -> torch.Tensor:
    """Spacing-aware first-order diffusion on physical displacement gradients."""
    with torch.amp.autocast(device_type=velocity.device.type, enabled=False):
        physical_velocity, spacing = normalized_velocity_to_physical(velocity, spacing_dhw)
        dz = (physical_velocity[:, :, 1:, :, :] - physical_velocity[:, :, :-1, :, :])
        dz = dz / spacing[:, 0].view(-1, 1, 1, 1, 1)
        dy = (physical_velocity[:, :, :, 1:, :] - physical_velocity[:, :, :, :-1, :])
        dy = dy / spacing[:, 1].view(-1, 1, 1, 1, 1)
        dx = (physical_velocity[:, :, :, :, 1:] - physical_velocity[:, :, :, :, :-1])
        dx = dx / spacing[:, 2].view(-1, 1, 1, 1, 1)
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
    with torch.amp.autocast(device_type=phi.device.type, enabled=False):
        vox = normalized_grid_to_voxel_grid(phi.float())
        d_dx = (vox[:, 1:-1, 1:-1, 2:, :] - vox[:, 1:-1, 1:-1, :-2, :]) * 0.5
        d_dy = (vox[:, 1:-1, 2:, 1:-1, :] - vox[:, 1:-1, :-2, 1:-1, :]) * 0.5
        d_dz = (vox[:, 2:, 1:-1, 1:-1, :] - vox[:, :-2, 1:-1, 1:-1, :]) * 0.5
        jac = torch.stack((d_dx, d_dy, d_dz), dim=-1)
        return torch.linalg.det(jac)


def jacobian_folding_penalty(
    phi_fwd: torch.Tensor,
    phi_inv: Optional[torch.Tensor] = None,
    minimum_determinant: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if minimum_determinant < 0.0:
        raise ValueError("minimum_determinant must be non-negative")

    transforms = (phi_fwd,) if phi_inv is None else (phi_fwd, phi_inv)
    penalties = []
    folding_ratios = []
    for transform in transforms:
        det = jacobian_determinant(transform)
        mean_square_violation = torch.relu(float(minimum_determinant) - det).square().mean()
        eps = mean_square_violation.new_tensor(1.0e-12)
        penalties.append(torch.sqrt(mean_square_violation + eps) - torch.sqrt(eps))
        folding_ratios.append((det <= 0).to(det.dtype).mean())

    penalty = torch.stack(penalties).mean()
    folding_ratio = torch.stack(folding_ratios).max()
    return penalty, folding_ratio
