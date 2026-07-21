from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import nn

from gam_reg.models.gaussian_types import GaussianMatchOutput, GaussianTokenBatch
from gam_reg.models.spatial_transformer import identity_grid


class GaussianToVolumePropagator(nn.Module):
    """Chunked anisotropic Gaussian-to-volume displacement propagation."""

    def __init__(self, token_chunk: int = 32, mahalanobis_clip: float = 30.0, eps: float = 1.0e-6):
        super().__init__()
        if token_chunk <= 0:
            raise ValueError("token_chunk must be positive")
        self.token_chunk = int(token_chunk)
        self.mahalanobis_clip = float(mahalanobis_clip)
        self.eps = float(eps)

    def forward(
        self,
        tokens: GaussianTokenBatch,
        match: GaussianMatchOutput,
        spatial_shape: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens.validate()
        match.validate()
        if match.displacement.shape[:2] != tokens.mu.shape[:2]:
            raise AssertionError("match displacement must align with moving tokens")
        d, h, w = int(spatial_shape[0]), int(spatial_shape[1]), int(spatial_shape[2])
        b, n, _ = tokens.mu.shape
        device = tokens.mu.device
        accum_dtype = torch.float32
        grid = identity_grid((d, h, w), batch_size=1, device=device, dtype=accum_dtype)
        grid = grid[:, None, :, :, :, :]
        mu = tokens.mu.float()
        cov_inv = torch.linalg.inv(tokens.cov.float())
        displacement = match.displacement.float()
        confidence = match.confidence.float().clamp(0.0, 1.0)

        num = torch.zeros((b, 3, d, h, w), device=device, dtype=accum_dtype)
        den = torch.zeros((b, 1, d, h, w), device=device, dtype=accum_dtype)
        for start in range(0, n, self.token_chunk):
            end = min(start + self.token_chunk, n)
            diff = grid - mu[:, start:end, None, None, None, :]
            inv = cov_inv[:, start:end]
            mahal = torch.einsum("bmdhwj,bmjk,bmdhwk->bmdhw", diff, inv, diff)
            mahal = mahal.clamp(0.0, self.mahalanobis_clip)
            weights = confidence[:, start:end, 0, None, None, None] * torch.exp(-0.5 * mahal)
            num = num + torch.einsum("bmdhw,bmc->bcdhw", weights, displacement[:, start:end, :])
            den = den + weights.sum(dim=1, keepdim=True)

        prior = num / (den + self.eps)
        conf = den / (1.0 + den)
        return prior.to(tokens.mu.dtype), conf.to(tokens.mu.dtype)
