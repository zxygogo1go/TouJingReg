import unittest

import torch

from gam_reg.losses.dice import dice_loss
from gam_reg.models.spatial_transformer import identity_grid


class AvailabilityAwareDiceLossTest(unittest.TestCase):
    def test_one_sided_missing_class_is_not_an_impossible_penalty(self):
        moving = torch.zeros(1, 3, 8, 8, 8)
        fixed = torch.zeros_like(moving)
        moving[:, 0] = 1.0
        fixed[:, 0] = 1.0
        moving[:, 1, 2:5, 2:5, 2:5] = 1.0
        moving[:, 0, 2:5, 2:5, 2:5] = 0.0
        fixed[:, 1, 2:5, 2:5, 2:5] = 1.0
        fixed[:, 0, 2:5, 2:5, 2:5] = 0.0
        fixed[:, 2, 5:7, 5:7, 5:7] = 1.0
        fixed[:, 0, 5:7, 5:7, 5:7] = 0.0
        loss = dice_loss(moving, fixed, identity_grid((8, 8, 8)))
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_no_shared_foreground_class_returns_differentiable_zero(self):
        moving = torch.zeros(1, 3, 8, 8, 8)
        fixed = torch.zeros_like(moving)
        moving[:, 1, 2:5, 2:5, 2:5] = 1.0
        fixed[:, 2, 2:5, 2:5, 2:5] = 1.0
        phi = identity_grid((8, 8, 8)).requires_grad_(True)
        loss = dice_loss(moving, fixed, phi)
        loss.backward()
        self.assertEqual(float(loss), 0.0)
        self.assertIsNotNone(phi.grad)
        self.assertTrue(torch.isfinite(phi.grad).all())

    def test_shared_misaligned_class_still_has_positive_loss(self):
        moving = torch.zeros(1, 3, 8, 8, 8)
        fixed = torch.zeros_like(moving)
        moving[:, 1, 1:3, 1:3, 1:3] = 1.0
        fixed[:, 1, 5:7, 5:7, 5:7] = 1.0
        loss = dice_loss(moving, fixed, identity_grid((8, 8, 8)))
        self.assertGreater(float(loss), 0.99)


if __name__ == "__main__":
    unittest.main()
