from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


def log_sinkhorn(
    cost: torch.Tensor,
    epsilon: float = 0.07,
    iterations: int = 30,
    convergence_tol: Optional[float] = 1.0e-4,
) -> torch.Tensor:
    """Balanced entropic OT in log-domain for uniform marginals."""
    if cost.ndim != 3:
        raise AssertionError("cost must have shape [B,Nm,Nf]")
    if not torch.is_floating_point(cost):
        raise AssertionError("cost must be floating point")
    b, nm, nf = cost.shape
    if nm <= 0 or nf <= 0:
        raise ValueError("cost must have non-empty token dimensions")
    eps = float(epsilon)
    if eps <= 0.0:
        raise ValueError("epsilon must be positive")

    log_k = (-cost.float() / eps).clamp_min(-1.0e6)
    log_r = -math.log(float(nm))
    log_c = -math.log(float(nf))
    u = cost.new_zeros((b, nm), dtype=torch.float32)
    v = cost.new_zeros((b, nf), dtype=torch.float32)
    for _ in range(int(iterations)):
        prev_u = u
        u = log_r - torch.logsumexp(log_k + v[:, None, :], dim=2)
        v = log_c - torch.logsumexp(log_k + u[:, :, None], dim=1)
        if convergence_tol is not None:
            delta = (u - prev_u).abs().max()
            if bool(delta < float(convergence_tol)):
                break
    log_p = log_k + u[:, :, None] + v[:, None, :]
    return torch.exp(log_p).to(cost.dtype)


class LogSinkhorn(nn.Module):
    def __init__(self, epsilon: float = 0.07, iterations: int = 30, convergence_tol: Optional[float] = 1.0e-4):
        super().__init__()
        self.epsilon = float(epsilon)
        self.iterations = int(iterations)
        self.convergence_tol = convergence_tol

    def forward(self, cost: torch.Tensor) -> torch.Tensor:
        return log_sinkhorn(cost, self.epsilon, self.iterations, self.convergence_tol)
