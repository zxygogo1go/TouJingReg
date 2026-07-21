import unittest

import torch

from gam_reg.losses.deformation_losses import jacobian_determinant
from gam_reg.models.diffeomorphic import DiffeomorphicIntegrator, constant_velocity
from gam_reg.models.spatial_transformer import compose_transforms, identity_grid


class DiffeomorphicIntegrationTest(unittest.TestCase):
    def test_zero_velocity_is_identity(self):
        v = torch.zeros(1, 3, 7, 8, 9)
        integrator = DiffeomorphicIntegrator(steps=5)
        phi_fwd, phi_inv = integrator(v)
        ident = identity_grid((7, 8, 9), batch_size=1)
        self.assertTrue(torch.allclose(phi_fwd, ident, atol=5e-6))
        self.assertTrue(torch.allclose(phi_inv, ident, atol=5e-6))

    def test_constant_translation_and_inverse_consistency(self):
        disp = torch.tensor([0.08, -0.04, 0.06])
        v = constant_velocity((9, 10, 11), disp, batch_size=1)
        integrator = DiffeomorphicIntegrator(steps=6)
        phi_fwd, phi_inv = integrator(v)
        ident = identity_grid((9, 10, 11), batch_size=1)
        interior = (slice(None), slice(2, -2), slice(2, -2), slice(2, -2), slice(None))
        self.assertTrue(torch.allclose(phi_fwd[interior], ident[interior] + disp, atol=2e-3))
        composed = compose_transforms(phi_fwd, phi_inv)
        self.assertTrue(torch.allclose(composed[interior], ident[interior], atol=4e-3))

    def test_identity_jacobian_is_one(self):
        ident = identity_grid((7, 8, 9), batch_size=1)
        det = jacobian_determinant(ident)
        self.assertTrue(torch.allclose(det, torch.ones_like(det), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
