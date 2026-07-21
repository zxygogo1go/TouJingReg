import unittest

import torch

from gam_reg.losses.deformation_losses import (
    jacobian_determinant,
    jacobian_folding_penalty,
    smoothness_loss,
)
from gam_reg.models.spatial_transformer import identity_grid
from gam_reg.metrics.jacobian_metrics import jacobian_metric_dict


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

    def test_physical_smoothness_has_shape_independent_scale(self):
        alpha = 0.1
        for shape, spacing in (((8, 9, 10), (3.0, 2.0, 1.0)), ((12, 13, 14), (1.0, 2.0, 4.0))):
            d, h, w = shape
            x = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, 1, w)
            velocity = torch.zeros(1, 3, d, h, w)
            velocity[:, 0:1] = alpha * x
            value = smoothness_loss(velocity, spacing_dhw=spacing)
            self.assertAlmostEqual(float(value), alpha * alpha / 9.0, places=5)

    def test_jacobian_floor_penalizes_near_singular_positive_transform(self):
        phi = identity_grid((8, 9, 10))
        phi[..., 0] = phi[..., 0] * 0.02
        penalty, folding = jacobian_folding_penalty(phi, minimum_determinant=0.05)
        self.assertGreater(float(penalty), 0.0)
        self.assertEqual(float(folding), 0.0)
        metrics = jacobian_metric_dict(phi, minimum_determinant=0.05)
        self.assertEqual(float(metrics["folding_ratio_metric"]), 0.0)
        self.assertGreater(float(metrics["below_minimum_det_j_ratio"]), 0.99)


if __name__ == "__main__":
    unittest.main()
