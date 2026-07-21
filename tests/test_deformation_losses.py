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

    def test_jacobian_penalty_uses_rms_not_volume_diluted_mean_square(self):
        phi = identity_grid((8, 9, 10))
        phi[..., 0] = phi[..., 0] * 0.02
        penalty, _ = jacobian_folding_penalty(
            phi,
            minimum_determinant=0.05,
            tail_weight=0.0,
        )
        self.assertAlmostEqual(float(penalty), 0.03, places=5)

    def test_jacobian_tail_penalty_emphasizes_sparse_deep_fold(self):
        phi = identity_grid((16, 16, 16))
        phi[:, 8, 8, 8, 0] = -1.0
        global_only, folding = jacobian_folding_penalty(
            phi,
            tail_weight=0.0,
        )
        with_tail, _ = jacobian_folding_penalty(
            phi,
            tail_fraction=0.001,
            tail_weight=0.25,
        )
        self.assertGreater(float(folding), 0.0)
        self.assertGreater(float(with_tail), float(global_only) * 2.0)

    def test_jacobian_tail_configuration_is_validated(self):
        identity = identity_grid((8, 9, 10))
        with self.assertRaisesRegex(ValueError, "tail_fraction"):
            jacobian_folding_penalty(identity, tail_fraction=1.1)
        with self.assertRaisesRegex(ValueError, "tail_weight"):
            jacobian_folding_penalty(identity, tail_weight=-0.1)

    def test_jacobian_penalty_and_metrics_cover_inverse_transform(self):
        identity = identity_grid((8, 9, 10))
        unsafe_inverse = identity.clone()
        unsafe_inverse[..., 0] = unsafe_inverse[..., 0] * -0.1
        penalty, folding = jacobian_folding_penalty(
            identity,
            phi_inv=unsafe_inverse,
            minimum_determinant=0.05,
        )
        metrics = jacobian_metric_dict(
            identity,
            minimum_determinant=0.05,
            phi_inv=unsafe_inverse,
        )
        self.assertGreater(float(penalty), 0.0)
        self.assertGreater(float(folding), 0.99)
        self.assertEqual(float(metrics["forward_folding_ratio"]), 0.0)
        self.assertGreater(float(metrics["inverse_folding_ratio"]), 0.99)
        self.assertLess(float(metrics["minimum_det_j"]), 0.0)

    def test_identity_jacobian_penalty_has_finite_zero_gradient(self):
        identity = identity_grid((8, 9, 10)).requires_grad_(True)
        penalty, _ = jacobian_folding_penalty(identity)
        penalty.backward()
        self.assertEqual(float(penalty), 0.0)
        self.assertIsNotNone(identity.grad)
        self.assertTrue(torch.isfinite(identity.grad).all())


if __name__ == "__main__":
    unittest.main()
