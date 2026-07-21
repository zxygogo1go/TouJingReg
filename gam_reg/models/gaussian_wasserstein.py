from __future__ import annotations

from typing import Tuple

import torch


def sqrtm_spd(matrix: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Matrix square root for symmetric positive definite matrices."""
    if matrix.shape[-2:] != (3, 3):
        raise AssertionError("matrix must have trailing shape [3,3]")
    eigvals, eigvecs = torch.linalg.eigh(matrix.float())
    eigvals = eigvals.clamp_min(float(eps))
    sqrt_diag = torch.diag_embed(eigvals.sqrt())
    sqrt_matrix = eigvecs @ sqrt_diag @ eigvecs.transpose(-1, -2)
    return sqrt_matrix.to(matrix.dtype)


@torch.cuda.amp.autocast(enabled=False)
def pairwise_gaussian_w2(
    mu_a: torch.Tensor,
    cov_a: torch.Tensor,
    mu_b: torch.Tensor,
    cov_b: torch.Tensor,
    eps: float = 1.0e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pairwise squared Gaussian 2-Wasserstein costs.

    Args:
        mu_a: [B,Na,3]
        cov_a: [B,Na,3,3]
        mu_b: [B,Nb,3]
        cov_b: [B,Nb,3,3]
    Returns:
        center_cost, cov_cost, w2_cost with shape [B,Na,Nb].
    """
    if mu_a.ndim != 3 or mu_b.ndim != 3 or mu_a.shape[-1] != 3 or mu_b.shape[-1] != 3:
        raise AssertionError("mu tensors must have shape [B,N,3]")
    if cov_a.shape[:2] != mu_a.shape[:2] or cov_b.shape[:2] != mu_b.shape[:2]:
        raise AssertionError("covariance batch/token dimensions must match mu")
    if cov_a.shape[-2:] != (3, 3) or cov_b.shape[-2:] != (3, 3):
        raise AssertionError("covariance tensors must have trailing shape [3,3]")
    if mu_a.shape[0] != mu_b.shape[0]:
        raise AssertionError("batch size mismatch")

    out_dtype = mu_a.dtype
    mu_a_f = mu_a.float()
    mu_b_f = mu_b.float()
    cov_a_f = cov_a.float()
    cov_b_f = cov_b.float()

    diff = mu_a_f[:, :, None, :] - mu_b_f[:, None, :, :]
    center_cost = diff.square().sum(dim=-1)

    sqrt_a = sqrtm_spd(cov_a_f, eps=eps)
    inner = sqrt_a[:, :, None] @ cov_b_f[:, None] @ sqrt_a[:, :, None]
    sqrt_inner = sqrtm_spd(inner, eps=eps)
    trace_a = cov_a_f.diagonal(dim1=-2, dim2=-1).sum(dim=-1)[:, :, None]
    trace_b = cov_b_f.diagonal(dim1=-2, dim2=-1).sum(dim=-1)[:, None, :]
    trace_inner = sqrt_inner.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cov_cost = (trace_a + trace_b - 2.0 * trace_inner).clamp_min(0.0)
    w2_cost = (center_cost + cov_cost).clamp_min(0.0)
    return center_cost.to(out_dtype), cov_cost.to(out_dtype), w2_cost.to(out_dtype)
