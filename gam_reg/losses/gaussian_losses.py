from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from gam_reg.models.gaussian_types import GaussianMatchOutput, GaussianTokenBatch
from gam_reg.models.spatial_transformer import sample_grid_at_points, spatial_transform


def gaussian_anchor_consistency_loss(
    tokens_moving: Dict[str, GaussianTokenBatch],
    matches: Dict[str, GaussianMatchOutput],
    phi_fwd: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    losses = []
    for key, tokens in tokens_moving.items():
        if key not in matches:
            continue
        match = matches[key]
        sampled_mu = sample_grid_at_points(phi_fwd, tokens.mu)
        weights = match.confidence.clamp(0.0, 1.0)
        loss = (weights * (sampled_mu - match.target_mu).abs()).sum() / (weights.sum() * 3.0 + eps)
        losses.append(loss)
    if not losses:
        return phi_fwd.sum() * 0.0
    return torch.stack(losses).sum()


def token_regularization_loss(tokens_by_scale: Dict[str, GaussianTokenBatch], kappa: float = 8.0, eps: float = 1.0e-6) -> torch.Tensor:
    losses = []
    for tokens in tokens_by_scale.values():
        offset = tokens.offset.square().sum(dim=-1).mean()
        ratio = tokens.sigma.max(dim=-1).values / (tokens.sigma.min(dim=-1).values + eps)
        cond = torch.relu(ratio - float(kappa)).square().mean()
        losses.append(offset + 0.1 * cond)
    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).sum()


def _sample_token_segmentation(seg: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    grid = mu[:, :, None, None, :]
    sampled = spatial_transform(seg.float(), grid, mode="bilinear", padding_mode="border")
    sampled = sampled[:, :, :, 0, 0].transpose(1, 2).contiguous()
    return sampled / (sampled.sum(dim=-1, keepdim=True) + 1.0e-6)


def anatomy_token_loss_for_batch(tokens: GaussianTokenBatch, seg: Optional[torch.Tensor]) -> torch.Tensor:
    if seg is None or tokens.anat_logits.shape[-1] == 0:
        return tokens.mu.sum() * 0.0
    if seg.shape[1] != tokens.anat_logits.shape[-1]:
        raise AssertionError("segmentation channel count must match anatomy logits")
    target = _sample_token_segmentation(seg, tokens.mu)
    log_prob = F.log_softmax(tokens.anat_logits, dim=-1)
    return -(target * log_prob).sum(dim=-1).mean()


def anatomy_token_loss(
    tokens_moving: Dict[str, GaussianTokenBatch],
    tokens_fixed: Dict[str, GaussianTokenBatch],
    moving_seg: Optional[torch.Tensor],
    fixed_seg: Optional[torch.Tensor],
) -> torch.Tensor:
    losses = []
    for tokens in tokens_moving.values():
        losses.append(anatomy_token_loss_for_batch(tokens, moving_seg))
    for tokens in tokens_fixed.values():
        losses.append(anatomy_token_loss_for_batch(tokens, fixed_seg))
    if not losses:
        if moving_seg is not None:
            return moving_seg.sum() * 0.0
        return torch.tensor(0.0)
    return torch.stack(losses).sum()
