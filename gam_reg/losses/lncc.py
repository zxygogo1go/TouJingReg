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
        c = fixed.shape[1]
        filt = torch.ones((c, 1) + self.window, device=fixed.device, dtype=fixed.dtype)
        padding = tuple(k // 2 for k in self.window)
        win_size = float(self.window[0] * self.window[1] * self.window[2])

        i = fixed
        j = warped_moving
        i_sum = F.conv3d(i, filt, padding=padding, groups=c)
        j_sum = F.conv3d(j, filt, padding=padding, groups=c)
        i2_sum = F.conv3d(i * i, filt, padding=padding, groups=c)
        j2_sum = F.conv3d(j * j, filt, padding=padding, groups=c)
        ij_sum = F.conv3d(i * j, filt, padding=padding, groups=c)

        u_i = i_sum / win_size
        u_j = j_sum / win_size
        cross = ij_sum - u_j * i_sum - u_i * j_sum + u_i * u_j * win_size
        i_var = (i2_sum - 2.0 * u_i * i_sum + u_i.square() * win_size).clamp_min(self.eps)
        j_var = (j2_sum - 2.0 * u_j * j_sum + u_j.square() * win_size).clamp_min(self.eps)
        cc = cross.square() / (i_var * j_var + self.eps)
        return -cc.mean()
