import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from gam_reg.data.dataset import VolumePairDataset, load_volume
from gam_reg.data.preprocessing import normalize_image_intensity


class ImageNormalizationTest(unittest.TestCase):
    def test_zero_one_maps_to_minus_one_one(self):
        volume = torch.tensor([0.0, 0.25, 0.5, 1.0])
        actual = normalize_image_intensity(volume, "zero_one")
        expected = torch.tensor([-1.0, -0.5, 0.0, 1.0])
        self.assertTrue(torch.allclose(actual, expected))

    def test_zero_one_rejects_wrong_range(self):
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            normalize_image_intensity(torch.tensor([0.0, 2.0]), "zero_one")

    def test_dataset_uses_selected_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "moving.npy", np.full((4, 4, 4), 0.25, dtype=np.float32))
            np.save(root / "fixed.npy", np.full((4, 4, 4), 0.75, dtype=np.float32))
            (root / "pairs.csv").write_text(
                "moving,fixed,moving_seg,fixed_seg\n"
                "moving.npy,fixed.npy,,\n",
                encoding="utf-8",
            )
            dataset = VolumePairDataset(
                root / "pairs.csv",
                data_root=root,
                image_normalization="zero_one",
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["moving"].shape), (1, 4, 4, 4))
            self.assertTrue(torch.allclose(sample["moving"], torch.full((1, 4, 4, 4), -0.5)))
            self.assertTrue(torch.allclose(sample["fixed"], torch.full((1, 4, 4, 4), 0.5)))
            self.assertTrue(torch.equal(sample["spacing_dhw"], torch.ones(3)))

    def test_load_volume_can_preserve_pre_normalized_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.npy"
            np.save(path, np.full((2, 2, 2), -0.25, dtype=np.float32))
            actual = load_volume(path, image_normalization="minus_one_one")
            self.assertTrue(torch.allclose(actual, torch.full((1, 2, 2, 2), -0.25)))


if __name__ == "__main__":
    unittest.main()
