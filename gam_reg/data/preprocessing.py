from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F


def ensure_channel_first(volume: torch.Tensor) -> torch.Tensor:
    """Convert [D,H,W] or [C,D,H,W] tensor to [C,D,H,W]."""
    if volume.ndim == 3:
        return volume.unsqueeze(0)
    if volume.ndim == 4:
        return volume
    if volume.ndim == 5 and volume.shape[0] == 1:
        return volume.squeeze(0)
    raise ValueError("volume must be [D,H,W], [C,D,H,W], or [1,C,D,H,W]")


def clip_normalize_ct(volume: torch.Tensor, clip_range: Sequence[float] = (-1000.0, 2000.0)) -> torch.Tensor:
    lo, hi = float(clip_range[0]), float(clip_range[1])
    volume = volume.float().clamp(lo, hi)
    return 2.0 * (volume - lo) / (hi - lo) - 1.0


def normalize_image_intensity(volume: torch.Tensor, mode: str = "hu") -> torch.Tensor:
    """Normalize an image to the model's expected intensity domain.

    ``hu`` clips raw CT values and maps them to [-1, 1]. ``zero_one`` is
    intended for preprocessed arrays already normalized to [0, 1].
    ``minus_one_one`` validates and preserves arrays already in [-1, 1].
    ``none`` leaves values unchanged for explicitly managed pipelines.
    """
    mode = str(mode).lower()
    volume = volume.float()
    if mode == "hu":
        return clip_normalize_ct(volume)
    if mode == "zero_one":
        if not torch.isfinite(volume).all():
            raise ValueError("zero_one image contains non-finite values")
        if float(volume.min()) < -1.0e-4 or float(volume.max()) > 1.0001:
            raise ValueError("zero_one image values must lie in [0, 1]")
        return volume.clamp(0.0, 1.0).mul(2.0).sub(1.0)
    if mode == "minus_one_one":
        if not torch.isfinite(volume).all():
            raise ValueError("minus_one_one image contains non-finite values")
        if float(volume.min()) < -1.0001 or float(volume.max()) > 1.0001:
            raise ValueError("minus_one_one image values must lie in [-1, 1]")
        return volume.clamp(-1.0, 1.0)
    if mode == "none":
        return volume
    raise ValueError("unknown image normalization mode: %s" % mode)


def crop_or_pad_3d(volume: torch.Tensor, target_shape: Sequence[int], value: float = 0.0) -> torch.Tensor:
    """Center crop/pad a [C,D,H,W] tensor to target D,H,W."""
    if volume.ndim != 4:
        raise AssertionError("volume must be [C,D,H,W]")
    target = tuple(int(v) for v in target_shape)
    c, d, h, w = volume.shape
    out = volume.new_full((c,) + target, float(value))
    src_slices = []
    dst_slices = []
    for src, tgt in zip((d, h, w), target):
        crop = min(src, tgt)
        src0 = (src - crop) // 2
        dst0 = (tgt - crop) // 2
        src_slices.append(slice(src0, src0 + crop))
        dst_slices.append(slice(dst0, dst0 + crop))
    out[(slice(None),) + tuple(dst_slices)] = volume[(slice(None),) + tuple(src_slices)]
    return out


def resize_volume(volume: torch.Tensor, target_shape: Sequence[int], mode: str = "trilinear") -> torch.Tensor:
    if volume.ndim != 4:
        raise AssertionError("volume must be [C,D,H,W]")
    align_corners = True if mode in {"trilinear", "bilinear"} else None
    kwargs = {"mode": mode}
    if align_corners is not None:
        kwargs["align_corners"] = align_corners
    return F.interpolate(volume.unsqueeze(0).float(), size=tuple(target_shape), **kwargs).squeeze(0)


def labels_to_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    if labels.ndim == 4 and labels.shape[0] == num_classes:
        return labels.float()
    if labels.ndim == 4 and labels.shape[0] == 1:
        labels = labels[0]
    if labels.ndim != 3:
        raise AssertionError("label volume must be [D,H,W] or [1,D,H,W]")
    one_hot = F.one_hot(labels.long().clamp_min(0), num_classes=int(num_classes))
    return one_hot.permute(3, 0, 1, 2).float()
