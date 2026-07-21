from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _window_tuple(window: Sequence[int]) -> Tuple[int, int, int]:
    if len(window) != 3:
        raise ValueError("lncc window must be [D,H,W]")
    return int(window[0]), int(window[1]), int(window[2])


class LNCCLoss(nn.Module):
    """Negative local normalized cross-correlation for 3D volumes."""

    def __init__(self, window: Sequence[int] = (9, 9, 9), eps: float = 1.0e-5):
        super().__init__()
        self.window = _window_tuple(window)
        self.eps = float(eps)

    def forward(self, fixed: torch.Tensor, warped_moving: torch.Tensor) -> torch.Tensor:
        if fixed.shape != warped_moving.shape or fixed.ndim != 5:
            raise AssertionError("fixed and warped_moving must both be [B,C,D,H,W] with identical shape")
        padding = tuple(k // 2 for k in self.window)
        device_type = fixed.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            i = fixed.float()
            j = warped_moving.float()

            def local_mean(value: torch.Tensor) -> torch.Tensor:
                return F.avg_pool3d(
                    value,
                    kernel_size=self.window,
                    stride=1,
                    padding=padding,
                    count_include_pad=True,
                )

            mean_i = local_mean(i)
            mean_j = local_mean(j)
            cross = local_mean(i * j) - mean_i * mean_j
            i_var = (local_mean(i.square()) - mean_i.square()).clamp_min(0.0)
            j_var = (local_mean(j.square()) - mean_j.square()).clamp_min(0.0)
            cc = cross.square() / (i_var * j_var + self.eps)
            cc = cc.clamp(0.0, 1.0)
            return -cc.mean()
