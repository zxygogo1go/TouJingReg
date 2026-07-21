from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from gam_reg.models.gaussian_types import GaussianTokenBatch


def make_anchor_grid(token_grid: Sequence[int], device=None, dtype=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return anchors [1,N,3] and cell size [1,1,3] in normalized xyz order."""
    if len(token_grid) != 3:
        raise ValueError("token_grid must be in Gd,Gh,Gw order")
    gd, gh, gw = int(token_grid[0]), int(token_grid[1]), int(token_grid[2])
    if min(gd, gh, gw) <= 0:
        raise ValueError("token_grid entries must be positive")
    z = torch.linspace(-1.0 + 1.0 / gd, 1.0 - 1.0 / gd, gd, device=device, dtype=dtype)
    y = torch.linspace(-1.0 + 1.0 / gh, 1.0 - 1.0 / gh, gh, device=device, dtype=dtype)
    x = torch.linspace(-1.0 + 1.0 / gw, 1.0 - 1.0 / gw, gw, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    anchors = torch.stack((xx, yy, zz), dim=-1).reshape(1, gd * gh * gw, 3)
    cell = torch.tensor([2.0 / gw, 2.0 / gh, 2.0 / gd], device=device, dtype=dtype).view(1, 1, 3)
    return anchors, cell


def rotation_6d_to_matrix(rotation_6d: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Continuous 6D representation converted to SO(3) with Gram-Schmidt."""
    if rotation_6d.shape[-1] != 6:
        raise AssertionError("rotation_6d last dimension must be 6")
    a1 = rotation_6d[..., 0:3]
    a2 = rotation_6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=eps)
    a2_orth = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(a2_orth, dim=-1, eps=eps)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


class GaussianAnatomyTokenizer(nn.Module):
    """Anchor-grid anisotropic Gaussian anatomy tokenizer."""

    def __init__(
        self,
        in_channels: int,
        token_dim: int = 96,
        token_grid: Sequence[int] = (4, 4, 4),
        sigma_min_ratio: float = 0.20,
        sigma_max_ratio: float = 1.20,
        offset_ratio: float = 0.35,
        cov_eps: float = 1.0e-5,
        use_anatomy_head: bool = False,
        num_anatomy_classes: int = 0,
        axis_sample_alpha: float = 0.75,
    ):
        super().__init__()
        self.token_grid = tuple(int(v) for v in token_grid)
        self.num_tokens = self.token_grid[0] * self.token_grid[1] * self.token_grid[2]
        self.token_dim = int(token_dim)
        self.sigma_min_ratio = float(sigma_min_ratio)
        self.sigma_max_ratio = float(sigma_max_ratio)
        self.offset_ratio = float(offset_ratio)
        self.cov_eps = float(cov_eps)
        self.use_anatomy_head = bool(use_anatomy_head)
        self.num_anatomy_classes = int(num_anatomy_classes) if self.use_anatomy_head else 0
        self.axis_sample_alpha = float(axis_sample_alpha)

        anchors, cell = make_anchor_grid(self.token_grid, dtype=torch.float32)
        self.register_buffer("anchor_mu", anchors, persistent=False)
        self.register_buffer("cell_size", cell, persistent=False)
        xi = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=torch.float32,
        )
        weights = torch.exp(-0.5 * (self.axis_sample_alpha * xi).square().sum(dim=-1))
        weights = weights / weights.sum()
        self.register_buffer("axis_xi", xi, persistent=False)
        self.register_buffer("axis_weights", weights.view(1, 1, 7, 1), persistent=False)

        self.feature_projection = nn.Conv3d(in_channels, self.token_dim, kernel_size=1)
        self.center_head = nn.Linear(self.token_dim, 3)
        self.scale_head = nn.Linear(self.token_dim, 3)
        self.rotation_head = nn.Linear(self.token_dim, 6)
        self.fusion_norm = nn.LayerNorm(self.token_dim)
        if self.use_anatomy_head:
            self.anatomy_head = nn.Linear(self.token_dim, self.num_anatomy_classes)
        else:
            self.anatomy_head = None

        nn.init.zeros_(self.center_head.weight)
        nn.init.zeros_(self.center_head.bias)
        nn.init.zeros_(self.scale_head.bias)
        with torch.no_grad():
            self.rotation_head.bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def _sample_axis_features(self, projected: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
        b, _, _, _, _ = projected.shape
        axes = rotation * sigma.unsqueeze(-2)
        offsets = torch.einsum("bnij,mj->bnmi", axes, self.axis_xi.to(projected))
        points = mu[:, :, None, :] + self.axis_sample_alpha * offsets
        grid = points.reshape(b, self.num_tokens, 7, 1, 3)
        sampled = F.grid_sample(
            projected,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sampled = sampled[:, :, :, :, 0].permute(0, 2, 3, 1).contiguous()
        return (sampled * self.axis_weights.to(sampled)).sum(dim=2)

    def forward(self, feature: torch.Tensor) -> GaussianTokenBatch:
        if feature.ndim != 5:
            raise AssertionError("feature must have shape [B,C,D,H,W]")
        b = feature.shape[0]
        projected = self.feature_projection(feature)
        pooled = F.adaptive_avg_pool3d(projected, self.token_grid)
        base = pooled.flatten(2).transpose(1, 2).contiguous()

        anchor_mu = self.anchor_mu.to(device=feature.device, dtype=feature.dtype)
        cell_size = self.cell_size.to(device=feature.device, dtype=feature.dtype)
        offset = self.offset_ratio * cell_size * torch.tanh(self.center_head(base))
        mu = (anchor_mu + offset).clamp(-1.0, 1.0)

        sigma_ratio = self.sigma_min_ratio + (self.sigma_max_ratio - self.sigma_min_ratio) * torch.sigmoid(self.scale_head(base))
        sigma = cell_size * sigma_ratio

        rotation = rotation_6d_to_matrix(self.rotation_head(base))
        diag = torch.diag_embed(sigma.square())
        cov = rotation @ diag @ rotation.transpose(-1, -2)
        eye = torch.eye(3, device=feature.device, dtype=feature.dtype).view(1, 1, 3, 3)
        cov = cov + self.cov_eps * eye

        sampled = self._sample_axis_features(projected, mu, sigma, rotation)
        fused = self.fusion_norm(base + sampled)
        token_feat = F.normalize(fused, p=2, dim=-1, eps=1.0e-6)
        if self.anatomy_head is not None:
            anat_logits = self.anatomy_head(fused)
        else:
            anat_logits = feature.new_zeros((b, self.num_tokens, 0))

        tokens = GaussianTokenBatch(mu=mu, sigma=sigma, rotation=rotation, cov=cov, feat=token_feat, anat_logits=anat_logits, offset=offset)
        tokens.validate()
        return tokens
