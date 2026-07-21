from __future__ import annotations

import torch

from gam_reg.models.spatial_transformer import sample_grid_at_points


def transform_landmarks(phi_fwd: torch.Tensor, moving_landmarks_xyz: torch.Tensor) -> torch.Tensor:
    return sample_grid_at_points(phi_fwd, moving_landmarks_xyz)


def target_registration_error(
    phi_fwd: torch.Tensor,
    moving_landmarks_xyz: torch.Tensor,
    fixed_landmarks_xyz: torch.Tensor,
    spacing_xyz: torch.Tensor | None = None,
) -> torch.Tensor:
    warped = transform_landmarks(phi_fwd, moving_landmarks_xyz)
    diff = warped - fixed_landmarks_xyz
    if spacing_xyz is not None:
        diff = diff * spacing_xyz.view(1, 1, 3)
    return diff.square().sum(dim=-1).sqrt().mean()
