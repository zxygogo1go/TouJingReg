from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from gam_reg.data.preprocessing import (
    clip_normalize_ct,
    crop_or_pad_3d,
    ensure_channel_first,
    labels_to_one_hot,
)
from gam_reg.models.spatial_transformer import identity_grid, spatial_transform


def _resolve_path(path: str, data_root: Optional[Path]) -> Path:
    p = Path(path)
    if not p.is_absolute() and data_root is not None:
        p = data_root / p
    return p


def _load_array(path: Path) -> np.ndarray:
    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(".npy"):
        return np.load(path)
    if suffix.endswith(".npz"):
        data = np.load(path)
        key = "volume" if "volume" in data else list(data.keys())[0]
        return data[key]
    if suffix.endswith(".pt") or suffix.endswith(".pth"):
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            for key in ("volume", "image", "array", "data"):
                if key in obj:
                    obj = obj[key]
                    break
        return obj.detach().cpu().numpy() if torch.is_tensor(obj) else np.asarray(obj)
    if suffix.endswith(".nii") or suffix.endswith(".nii.gz"):
        import nibabel as nib

        return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
    if suffix.endswith(".nrrd") or suffix.endswith(".seg.nrrd"):
        import SimpleITK as sitk

        image = sitk.ReadImage(str(path))
        return sitk.GetArrayFromImage(image)
    raise ValueError("unsupported volume extension: %s" % path)


def load_volume(
    path: str | Path,
    is_seg: bool = False,
    num_classes: Optional[int] = None,
    normalize_image: bool = True,
    target_shape: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    arr = _load_array(Path(path))
    tensor = torch.from_numpy(np.asarray(arr))
    if is_seg:
        if num_classes is None:
            tensor = ensure_channel_first(tensor.float())
        else:
            tensor = labels_to_one_hot(tensor, int(num_classes))
        if target_shape is not None:
            tensor = crop_or_pad_3d(tensor, target_shape)
        return tensor.float()
    tensor = ensure_channel_first(tensor.float())
    if normalize_image:
        tensor = clip_normalize_ct(tensor)
    if target_shape is not None:
        tensor = crop_or_pad_3d(tensor, target_shape)
    return tensor.float()


class VolumePairDataset(Dataset):
    """Manifest-driven paired volume dataset.

    CSV columns: moving,fixed and optional moving_seg,fixed_seg. Paths are
    resolved relative to data_root when not absolute.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        data_root: Optional[str | Path] = None,
        num_seg_classes: Optional[int] = None,
        normalize_images: bool = True,
        target_shape: Optional[Sequence[int]] = None,
    ):
        self.manifest_csv = Path(manifest_csv)
        self.data_root = None if data_root is None else Path(data_root)
        self.num_seg_classes = num_seg_classes
        self.normalize_images = bool(normalize_images)
        self.target_shape = target_shape
        with self.manifest_csv.open("r", newline="", encoding="utf-8") as f:
            self.rows: List[Dict[str, str]] = list(csv.DictReader(f))
        if not self.rows:
            raise ValueError("manifest is empty: %s" % self.manifest_csv)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.rows[int(index)]
        moving_path = _resolve_path(row["moving"], self.data_root)
        fixed_path = _resolve_path(row["fixed"], self.data_root)
        sample: Dict[str, torch.Tensor] = {
            "moving": load_volume(moving_path, normalize_image=self.normalize_images, target_shape=self.target_shape),
            "fixed": load_volume(fixed_path, normalize_image=self.normalize_images, target_shape=self.target_shape),
        }
        if row.get("moving_seg") and row.get("fixed_seg"):
            sample["moving_seg"] = load_volume(
                _resolve_path(row["moving_seg"], self.data_root),
                is_seg=True,
                num_classes=self.num_seg_classes,
                normalize_image=False,
                target_shape=self.target_shape,
            )
            sample["fixed_seg"] = load_volume(
                _resolve_path(row["fixed_seg"], self.data_root),
                is_seg=True,
                num_classes=self.num_seg_classes,
                normalize_image=False,
                target_shape=self.target_shape,
            )
        return sample


class SyntheticRegistrationDataset(Dataset):
    """Small reproducible synthetic 3D registration pairs for smoke/warm-up."""

    def __init__(
        self,
        num_samples: int = 128,
        spatial_shape: Sequence[int] = (32, 40, 32),
        num_blobs: int = 6,
        max_translation: float = 0.18,
        seed: int = 1234,
    ):
        self.num_samples = int(num_samples)
        self.spatial_shape = tuple(int(v) for v in spatial_shape)
        self.num_blobs = int(num_blobs)
        self.max_translation = float(max_translation)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(self.seed + int(index))
        grid = identity_grid(self.spatial_shape, batch_size=1)
        xyz = grid[0]
        volume = torch.zeros(self.spatial_shape)
        for _ in range(self.num_blobs):
            center = torch.empty(3).uniform_(-0.65, 0.65, generator=gen)
            sigma = torch.empty(3).uniform_(0.12, 0.35, generator=gen)
            amp = torch.empty(1).uniform_(0.5, 1.0, generator=gen)
            diff = (xyz - center.view(1, 1, 1, 3)) / sigma.view(1, 1, 1, 3)
            volume = volume + amp[0] * torch.exp(-0.5 * diff.square().sum(dim=-1))
        fixed = volume.unsqueeze(0)
        fixed = 2.0 * (fixed - fixed.min()) / (fixed.max() - fixed.min() + 1.0e-6) - 1.0
        disp = torch.empty(3).uniform_(-self.max_translation, self.max_translation, generator=gen)
        sample_grid = grid + disp.view(1, 1, 1, 1, 3)
        moving = spatial_transform(fixed.unsqueeze(0), sample_grid).squeeze(0)
        return {"moving": moving.float(), "fixed": fixed.float()}
