from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from gam_reg.models.gaussian_types import GaussianMatchOutput, GaussianTokenBatch
from gam_reg.models.gaussian_wasserstein import pairwise_gaussian_w2
from gam_reg.models.sinkhorn import log_sinkhorn


def _normalize_cost(cost: torch.Tensor) -> torch.Tensor:
    return cost / (cost.detach().mean(dim=(-2, -1), keepdim=True) + 1.0e-6)


class LogSinkhornMatcher(nn.Module):
    """Gaussian W2 + feature/anatomy cost matcher with log-domain Sinkhorn."""

    def __init__(
        self,
        lambda_center: float = 1.0,
        lambda_covariance: float = 0.5,
        lambda_feature: float = 1.0,
        lambda_anatomy: float = 0.2,
        sinkhorn_epsilon: float = 0.07,
        sinkhorn_iterations: int = 30,
        convergence_tol: Optional[float] = 1.0e-4,
        spatial_radius: Optional[float] = None,
        large_cost: float = 1.0e4,
        use_sinkhorn: bool = True,
    ):
        super().__init__()
        self.lambda_center = float(lambda_center)
        self.lambda_covariance = float(lambda_covariance)
        self.lambda_feature = float(lambda_feature)
        self.lambda_anatomy = float(lambda_anatomy)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.convergence_tol = convergence_tol
        self.spatial_radius = None if spatial_radius is None else float(spatial_radius)
        self.large_cost = float(large_cost)
        self.use_sinkhorn = bool(use_sinkhorn)

    def build_cost(self, moving_tokens: GaussianTokenBatch, fixed_tokens: GaussianTokenBatch) -> torch.Tensor:
        moving_tokens.validate()
        fixed_tokens.validate()
        center_cost, cov_cost, _ = pairwise_gaussian_w2(
            moving_tokens.mu,
            moving_tokens.cov,
            fixed_tokens.mu,
            fixed_tokens.cov,
        )
        feature_cost = 1.0 - torch.matmul(moving_tokens.feat, fixed_tokens.feat.transpose(1, 2)).clamp(-1.0, 1.0)
        cost = (
            self.lambda_center * _normalize_cost(center_cost)
            + self.lambda_covariance * _normalize_cost(cov_cost)
            + self.lambda_feature * feature_cost
        )
        if (
            self.lambda_anatomy > 0.0
            and moving_tokens.anat_logits.shape[-1] > 0
            and fixed_tokens.anat_logits.shape[-1] > 0
        ):
            pm = moving_tokens.anat_logits.softmax(dim=-1)
            pf = fixed_tokens.anat_logits.softmax(dim=-1)
            anatomy_cost = 1.0 - torch.matmul(pm, pf.transpose(1, 2)).clamp(0.0, 1.0)
            cost = cost + self.lambda_anatomy * anatomy_cost
        if self.spatial_radius is not None:
            center_distance = center_cost.clamp_min(0.0).sqrt()
            mask = center_distance < self.spatial_radius
            cost = cost.masked_fill(~mask, self.large_cost)
        return cost

    def forward(self, moving_tokens: GaussianTokenBatch, fixed_tokens: GaussianTokenBatch) -> GaussianMatchOutput:
        cost = self.build_cost(moving_tokens, fixed_tokens)
        _, nm, nf = cost.shape
        if self.use_sinkhorn:
            transport = log_sinkhorn(
                cost,
                epsilon=self.sinkhorn_epsilon,
                iterations=self.sinkhorn_iterations,
                convergence_tol=self.convergence_tol,
            )
            row_prob = transport / (transport.sum(dim=-1, keepdim=True) + 1.0e-8)
        else:
            row_prob = torch.softmax(-cost / self.sinkhorn_epsilon, dim=-1)
            transport = row_prob / float(nm)

        target_mu = row_prob @ fixed_tokens.mu
        displacement = target_mu - moving_tokens.mu
        entropy = -(row_prob * torch.log(row_prob + 1.0e-8)).sum(dim=-1, keepdim=True)
        if nf > 1:
            confidence = 1.0 - entropy / math.log(float(nf))
        else:
            confidence = torch.ones_like(entropy)
        confidence = confidence.clamp(0.0, 1.0)
        output = GaussianMatchOutput(
            transport=transport,
            row_prob=row_prob,
            target_mu=target_mu,
            displacement=displacement,
            confidence=confidence,
            cost=cost,
        )
        output.validate()
        return output
