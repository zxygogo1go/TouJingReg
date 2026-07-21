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
    phi_inv: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    def metrics(transform: torch.Tensor) -> Dict[str, torch.Tensor]:
        det = jacobian_determinant(transform)
        return {
            "folding_ratio": (det <= 0).to(det.dtype).mean(),
            "below_minimum_det_j_ratio": (det < float(minimum_determinant)).to(det.dtype).mean(),
            "mean_abs_det_j_minus_one": (det - 1.0).abs().mean(),
            "minimum_det_j": det.min(),
        }

    forward = metrics(phi_fwd)
    if phi_inv is None:
        return {
            "folding_ratio_metric": forward["folding_ratio"],
            "below_minimum_det_j_ratio": forward["below_minimum_det_j_ratio"],
            "mean_abs_det_j_minus_one": forward["mean_abs_det_j_minus_one"],
            "minimum_det_j": forward["minimum_det_j"],
        }

    inverse = metrics(phi_inv)
    result = {
        "folding_ratio_metric": torch.maximum(forward["folding_ratio"], inverse["folding_ratio"]),
        "below_minimum_det_j_ratio": torch.maximum(
            forward["below_minimum_det_j_ratio"], inverse["below_minimum_det_j_ratio"]
        ),
        "mean_abs_det_j_minus_one": 0.5
        * (forward["mean_abs_det_j_minus_one"] + inverse["mean_abs_det_j_minus_one"]),
        "minimum_det_j": torch.minimum(forward["minimum_det_j"], inverse["minimum_det_j"]),
    }
    for prefix, values in (("forward", forward), ("inverse", inverse)):
        for key, value in values.items():
            result[prefix + "_" + key] = value
    return result
