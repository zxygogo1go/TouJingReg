import unittest

import torch

from gam_reg.losses.lncc import LNCCLoss


class LNCCTest(unittest.TestCase):
    def test_identical_random_volumes_are_near_negative_one(self):
        torch.manual_seed(7)
        image = torch.rand(1, 1, 16, 17, 18) * 2.0 - 1.0
        value = LNCCLoss((5, 5, 5))(image, image)
        self.assertGreaterEqual(float(value), -1.000001)
        self.assertLess(float(value), -0.99)

    def test_low_variance_inputs_remain_bounded_and_differentiable(self):
        torch.manual_seed(11)
        fixed = -1.0 + 1.0e-4 * torch.randn(1, 1, 16, 16, 16)
        moving = (-1.0 + 1.0e-4 * torch.randn(1, 1, 16, 16, 16)).requires_grad_(True)
        value = LNCCLoss((9, 9, 9))(fixed, moving)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertGreaterEqual(float(value), -1.000001)
        self.assertLessEqual(float(value), 0.0)
        self.assertIsNotNone(moving.grad)
        self.assertTrue(torch.isfinite(moving.grad).all())

    def test_autocast_input_is_computed_in_float32(self):
        fixed = torch.rand(1, 1, 12, 12, 12)
        moving = torch.rand(1, 1, 12, 12, 12, requires_grad=True)
        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            value = LNCCLoss((5, 5, 5))(fixed, moving)
        self.assertEqual(value.dtype, torch.float32)
        self.assertGreaterEqual(float(value), -1.000001)
        self.assertLessEqual(float(value), 0.0)


if __name__ == "__main__":
    unittest.main()
