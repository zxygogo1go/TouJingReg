from __future__ import annotations

import random
from typing import Dict, Optional

import torch


def random_intensity_jitter(volume: torch.Tensor, gamma_range=(0.85, 1.15), shift_range=(-0.1, 0.1), noise_std=0.02) -> torch.Tensor:
    gamma = random.uniform(float(gamma_range[0]), float(gamma_range[1]))
    shift = random.uniform(float(shift_range[0]), float(shift_range[1]))
    x = ((volume + 1.0) * 0.5).clamp(0.0, 1.0).pow(gamma)
    x = 2.0 * x - 1.0 + shift
    if noise_std > 0:
        x = x + torch.randn_like(x) * float(noise_std)
    return x.clamp(-1.0, 1.0)


def random_synchronous_flip(sample: Dict[str, torch.Tensor], p: float = 0.5) -> Dict[str, torch.Tensor]:
    """Flip moving/fixed and optional segmentations together along random spatial axes."""
    out = dict(sample)
    for axis in (1, 2, 3):
        if random.random() < p:
            for key in ("moving", "fixed", "moving_seg", "fixed_seg"):
                if key in out and out[key] is not None:
                    out[key] = torch.flip(out[key], dims=(axis,))
    return out
