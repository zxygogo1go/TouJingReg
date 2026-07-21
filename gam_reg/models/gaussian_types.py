from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GaussianTokenBatch:
    mu: torch.Tensor
    sigma: torch.Tensor
    rotation: torch.Tensor
    cov: torch.Tensor
    feat: torch.Tensor
    anat_logits: torch.Tensor
    offset: torch.Tensor

    def validate(self) -> None:
        if self.mu.ndim != 3 or self.mu.shape[-1] != 3:
            raise AssertionError("mu must have shape [B,N,3]")
        b, n, _ = self.mu.shape
        expected_bn = (b, n)
        if tuple(self.sigma.shape[:2]) != expected_bn or self.sigma.shape[-1] != 3:
            raise AssertionError("sigma must have shape [B,N,3]")
        if tuple(self.rotation.shape) != (b, n, 3, 3):
            raise AssertionError("rotation must have shape [B,N,3,3]")
        if tuple(self.cov.shape) != (b, n, 3, 3):
            raise AssertionError("cov must have shape [B,N,3,3]")
        if self.feat.ndim != 3 or tuple(self.feat.shape[:2]) != expected_bn:
            raise AssertionError("feat must have shape [B,N,Ct]")
        if self.anat_logits.ndim != 3 or tuple(self.anat_logits.shape[:2]) != expected_bn:
            raise AssertionError("anat_logits must have shape [B,N,Ka]")
        if tuple(self.offset.shape) != (b, n, 3):
            raise AssertionError("offset must have shape [B,N,3]")


@dataclass
class GaussianMatchOutput:
    transport: torch.Tensor
    row_prob: torch.Tensor
    target_mu: torch.Tensor
    displacement: torch.Tensor
    confidence: torch.Tensor
    cost: torch.Tensor

    def validate(self) -> None:
        if self.transport.ndim != 3:
            raise AssertionError("transport must have shape [B,Nm,Nf]")
        b, nm, nf = self.transport.shape
        if tuple(self.row_prob.shape) != (b, nm, nf):
            raise AssertionError("row_prob must have shape [B,Nm,Nf]")
        if tuple(self.target_mu.shape) != (b, nm, 3):
            raise AssertionError("target_mu must have shape [B,Nm,3]")
        if tuple(self.displacement.shape) != (b, nm, 3):
            raise AssertionError("displacement must have shape [B,Nm,3]")
        if tuple(self.confidence.shape) != (b, nm, 1):
            raise AssertionError("confidence must have shape [B,Nm,1]")
        if tuple(self.cost.shape) != (b, nm, nf):
            raise AssertionError("cost must have shape [B,Nm,Nf]")
