from __future__ import annotations

from typing import Any, Iterable, List, Tuple

import torch


def make_grad_scaler(
    enabled: bool,
    init_scale: float = 1024.0,
    growth_interval: int = 2000,
) -> Any:
    """Create a CUDA GradScaler across the PyTorch 2.x API transition."""
    kwargs = {
        "enabled": bool(enabled),
        "init_scale": float(init_scale),
        "growth_interval": int(growth_interval),
    }
    amp_grad_scaler = getattr(torch.amp, "GradScaler", None)
    if amp_grad_scaler is not None:
        try:
            return amp_grad_scaler("cuda", **kwargs)
        except TypeError:
            return amp_grad_scaler(**kwargs)
    return torch.cuda.amp.GradScaler(**kwargs)


def require_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value.detach()).all()):
        raise FloatingPointError("non-finite %s detected" % name)


def nonfinite_gradient_names(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
) -> List[str]:
    names = []
    for name, parameter in named_parameters:
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad.detach()).all()):
            names.append(name)
    return names
