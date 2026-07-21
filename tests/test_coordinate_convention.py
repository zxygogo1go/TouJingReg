import unittest

import torch

from gam_reg.models.spatial_transformer import identity_grid, spatial_transform


class CoordinateConventionTest(unittest.TestCase):
    def test_grid_last_dimension_controls_xyz(self):
        grid = identity_grid((3, 4, 5), batch_size=1)
        self.assertEqual(tuple(grid.shape), (1, 3, 4, 5, 3))
        self.assertTrue(torch.allclose(grid[0, 1, 2, :, 0], torch.linspace(-1, 1, 5)))
        self.assertTrue(torch.allclose(grid[0, 1, :, 2, 1], torch.linspace(-1, 1, 4)))
        self.assertTrue(torch.allclose(grid[0, :, 2, 2, 2], torch.linspace(-1, 1, 3)))

    def test_inverse_grid_warps_moving_to_fixed_direction(self):
        d, h, w = 5, 5, 5
        x_ramp = identity_grid((d, h, w), batch_size=1)[..., 0]
        moving = x_ramp[:, None]
        identity = identity_grid((d, h, w), batch_size=1)
        step_x = 2.0 / (w - 1)
        phi_inv = identity.clone()
        phi_inv[..., 0] = phi_inv[..., 0] - step_x
        warped = spatial_transform(moving, phi_inv)

        # The fixed voxel at x-index 2 samples moving x-index 1 after inverse shift.
        self.assertAlmostEqual(
            float(warped[0, 0, 2, 2, 2]),
            float(moving[0, 0, 2, 2, 1]),
            places=5,
        )
        self.assertEqual(tuple(warped.shape[-3:]), (d, h, w))


if __name__ == "__main__":
    unittest.main()
