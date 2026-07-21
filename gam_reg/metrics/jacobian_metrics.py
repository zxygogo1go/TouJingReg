from __future__ import annotations

import torch
from typing import Dict

from gam_reg.losses.deformation_losses import jacobian_determinant


def folding_ratio(phi_fwd: torch.Tensor) -> torch.Tensor:
    det = jacobian_determinant(phi_fwd)
    return (det <= 0).to(det.dtype).mean()


def mean_abs_det_j_minus_one(phi_fwd: torch.Tensor) -> torch.Tensor:
    det = jacobian_determinant(phi_fwd)
    return (det - 1.0).abs().mean()


def jacobian_metric_dict(
    phi_fwd: torch.Tensor,
    minimum_determinant: float = 0.05,
) -> Dict[str, torch.Tensor]:
    det = jacobian_determinant(phi_fwd)
    return {
        "folding_ratio_metric": (det <= 0).to(det.dtype).mean(),
        "below_minimum_det_j_ratio": (det < float(minimum_determinant)).to(det.dtype).mean(),
        "mean_abs_det_j_minus_one": (det - 1.0).abs().mean(),
        "minimum_det_j": det.min(),
    }
