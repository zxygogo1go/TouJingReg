from __future__ import annotations

import torch

from gam_reg.losses.deformation_losses import jacobian_determinant


def folding_ratio(phi_fwd: torch.Tensor) -> torch.Tensor:
    det = jacobian_determinant(phi_fwd)
    return (det <= 0).to(det.dtype).mean()


def mean_abs_det_j_minus_one(phi_fwd: torch.Tensor) -> torch.Tensor:
    det = jacobian_determinant(phi_fwd)
    return (det - 1.0).abs().mean()
