import unittest

import torch

from gam_reg.losses.deformation_losses import (
    jacobian_determinant,
    jacobian_folding_penalty,
    smoothness_loss,
)
from gam_reg.models.spatial_transformer import identity_grid


class DeformationLossPrecisionTest(unittest.TestCase):
    def test_autocast_identity_has_unit_jacobian_and_no_folding(self):
        identity = identity_grid((8, 9, 10))
        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            determinant = jacobian_determinant(identity)
            penalty, folding = jacobian_folding_penalty(identity)
        self.assertEqual(determinant.dtype, torch.float32)
        self.assertTrue(torch.allclose(determinant, torch.ones_like(determinant), atol=1.0e-5))
        self.assertEqual(float(penalty), 0.0)
        self.assertEqual(float(folding), 0.0)

    def test_autocast_smoothness_is_float32_and_differentiable(self):
        velocity = (torch.randn(1, 3, 8, 9, 10) * 1.0e-3).requires_grad_(True)
        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            value = smoothness_loss(velocity)
        value.backward()
        self.assertEqual(value.dtype, torch.float32)
        self.assertTrue(torch.isfinite(value))
        self.assertIsNotNone(velocity.grad)
        self.assertTrue(torch.isfinite(velocity.grad).all())


if __name__ == "__main__":
    unittest.main()
