from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
import torch.nn.functional as F

from gam_reg.models.spatial_transformer import resize_grid, spatial_transform


def multi_scale_feature_similarity_loss(
    moving_features: List[torch.Tensor],
    fixed_features: List[torch.Tensor],
    phi_inv: torch.Tensor,
    levels: Sequence[int] = (2, 3),
) -> torch.Tensor:
    if len(moving_features) != len(fixed_features):
        raise AssertionError("moving/fixed feature lists must have the same length")
    losses = []
    for level in levels:
        fm = moving_features[int(level)]
        ff = fixed_features[int(level)]
        if fm.shape != ff.shape:
            raise AssertionError("feature shapes must match")
        grid_l = resize_grid(phi_inv, fm.shape[-3:])
        warped = spatial_transform(fm, grid_l)
        warped_n = F.normalize(warped, p=2, dim=1, eps=1.0e-6)
        fixed_n = F.normalize(ff, p=2, dim=1, eps=1.0e-6)
        cosine = (warped_n * fixed_n).sum(dim=1)
        losses.append((1.0 - cosine).mean())
    if not losses:
        return phi_inv.sum() * 0.0
    return torch.stack(losses).sum()
