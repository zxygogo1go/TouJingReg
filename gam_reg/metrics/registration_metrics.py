from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from gam_reg.losses.dice import per_class_dice
from gam_reg.models.spatial_transformer import spatial_transform


def mean_squared_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).square().mean()


def mean_absolute_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).abs().mean()


def dice_per_class_after_warp(moving_seg: torch.Tensor, fixed_seg: torch.Tensor, phi_inv: torch.Tensor) -> torch.Tensor:
    return per_class_dice(moving_seg, fixed_seg, phi_inv)


def available_hard_dice_per_class(
    moving_seg: torch.Tensor,
    fixed_seg: torch.Tensor,
    phi_inv: Optional[torch.Tensor] = None,
    exclude_background: bool = True,
    eps: float = 1.0e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return nearest-neighbor Dice and a mask of classes present at both timepoints."""
    if moving_seg.shape != fixed_seg.shape or moving_seg.ndim != 5:
        raise AssertionError("segmentations must be one-hot tensors [B,K,D,H,W] with identical shape")
    if phi_inv is not None:
        warped = spatial_transform(
            moving_seg.float(),
            phi_inv,
            mode="nearest",
            padding_mode="border",
        )
    else:
        warped = moving_seg.float()

    num_classes = int(moving_seg.shape[1])
    warped_labels = warped.argmax(dim=1)
    fixed_labels = fixed_seg.argmax(dim=1)
    warped_hard = F.one_hot(warped_labels, num_classes=num_classes).movedim(-1, 1).float()
    fixed_hard = F.one_hot(fixed_labels, num_classes=num_classes).movedim(-1, 1).float()
    moving_presence = moving_seg.float()
    fixed_presence = fixed_seg.float()
    if exclude_background and num_classes > 1:
        warped_hard = warped_hard[:, 1:]
        fixed_hard = fixed_hard[:, 1:]
        moving_presence = moving_presence[:, 1:]
        fixed_presence = fixed_presence[:, 1:]

    dims = (2, 3, 4)
    intersection = (warped_hard * fixed_hard).sum(dim=dims)
    denominator = warped_hard.sum(dim=dims) + fixed_hard.sum(dim=dims)
    dice = (2.0 * intersection + float(eps)) / (denominator + float(eps))
    available = (moving_presence.sum(dim=dims) > 0) & (fixed_presence.sum(dim=dims) > 0)
    return dice, available


def registration_dice_metric_dict(
    moving_seg: torch.Tensor,
    fixed_seg: torch.Tensor,
    phi_inv: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    before, available = available_hard_dice_per_class(moving_seg, fixed_seg)
    after, after_available = available_hard_dice_per_class(
        moving_seg,
        fixed_seg,
        phi_inv=phi_inv,
    )
    if not torch.equal(available, after_available):
        raise AssertionError("Dice class availability changed during label propagation")

    metrics: Dict[str, torch.Tensor] = {}
    if bool(available.any()):
        metrics["dice_score_before"] = before[available].mean()
        metrics["dice_score_after"] = after[available].mean()
        metrics["dice_score_gain"] = metrics["dice_score_after"] - metrics["dice_score_before"]
    for class_offset in range(before.shape[1]):
        valid = available[:, class_offset]
        if bool(valid.any()):
            class_id = class_offset + 1
            before_value = before[:, class_offset][valid].mean()
            after_value = after[:, class_offset][valid].mean()
            metrics["dice_score_class_%d_before" % class_id] = before_value
            metrics["dice_score_class_%d_after" % class_id] = after_value
            metrics["dice_score_class_%d_gain" % class_id] = after_value - before_value
    return metrics
