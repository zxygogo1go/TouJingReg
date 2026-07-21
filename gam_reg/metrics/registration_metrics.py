from __future__ import annotations

import torch

from gam_reg.losses.dice import per_class_dice


def mean_squared_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).square().mean()


def mean_absolute_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).abs().mean()


def dice_per_class_after_warp(moving_seg: torch.Tensor, fixed_seg: torch.Tensor, phi_inv: torch.Tensor) -> torch.Tensor:
    return per_class_dice(moving_seg, fixed_seg, phi_inv)
