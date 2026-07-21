from __future__ import annotations

from typing import Any

import torch


def make_grad_scaler(enabled: bool) -> Any:
    """Create a CUDA GradScaler across the PyTorch 2.x API transition."""
    amp_grad_scaler = getattr(torch.amp, "GradScaler", None)
    if amp_grad_scaler is not None:
        try:
            return amp_grad_scaler("cuda", enabled=bool(enabled))
        except TypeError:
            return amp_grad_scaler(enabled=bool(enabled))
    return torch.cuda.amp.GradScaler(enabled=bool(enabled))


def require_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value.detach()).all()):
        raise FloatingPointError("non-finite %s detected" % name)
