from __future__ import annotations

from typing import Optional

import torch

from gam_reg.models.spatial_transformer import spatial_transform


def dice_loss(
    moving_seg: Optional[torch.Tensor],
    fixed_seg: Optional[torch.Tensor],
    phi_inv: torch.Tensor,
    exclude_background: bool = True,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    if moving_seg is None or fixed_seg is None:
        return phi_inv.sum() * 0.0
    if moving_seg.shape != fixed_seg.shape or moving_seg.ndim != 5:
        raise AssertionError("segmentations must be one-hot tensors [B,K,D,H,W] with identical shape")
    warped = spatial_transform(moving_seg.float(), phi_inv, mode="bilinear", padding_mode="border")
    fixed = fixed_seg.float()
    moving_presence = moving_seg.float()
    if exclude_background and warped.shape[1] > 1:
        warped = warped[:, 1:]
        fixed = fixed[:, 1:]
        moving_presence = moving_presence[:, 1:]
    dims = (2, 3, 4)
    inter = (warped * fixed).sum(dim=dims)
    denom = warped.sum(dim=dims) + fixed.sum(dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    available = (moving_presence.sum(dim=dims) > eps) & (fixed.sum(dim=dims) > eps)
    if not bool(available.any()):
        return phi_inv.sum() * 0.0
    return (1.0 - dice)[available].mean()


def per_class_dice(
    moving_seg: torch.Tensor,
    fixed_seg: torch.Tensor,
    phi_inv: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    warped = spatial_transform(moving_seg.float(), phi_inv, mode="bilinear", padding_mode="border")
    fixed = fixed_seg.float()
    dims = (0, 2, 3, 4)
    inter = (warped * fixed).sum(dim=dims)
    denom = warped.sum(dim=dims) + fixed.sum(dim=dims)
    return (2.0 * inter + eps) / (denom + eps)
