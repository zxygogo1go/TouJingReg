import unittest

import torch

from gam_reg.losses.total_loss import TotalRegistrationLoss
from gam_reg.models.gam_reg import GAMReg
from tests.test_model_forward import tiny_config


class ModelSmokeTest(unittest.TestCase):
    def test_small_volume_forward_loss_backward(self):
        torch.manual_seed(13)
        cfg = tiny_config()
        model = GAMReg(cfg)
        loss_fn = TotalRegistrationLoss(cfg)
        moving = torch.randn(1, 1, 32, 40, 32)
        fixed = torch.randn(1, 1, 32, 40, 32)
        out = model(moving, fixed, return_debug=True)
        self.assertEqual(tuple(out["warped_moving"].shape), (1, 1, 32, 40, 32))
        self.assertEqual(tuple(out["velocity"].shape), (1, 3, 32, 40, 32))
        self.assertEqual(tuple(out["phi_inv"].shape), (1, 32, 40, 32, 3))
        total, components = loss_fn(out, fixed=fixed)
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertFalse(torch.isnan(out["velocity"]).any())
        self.assertFalse(torch.isinf(out["velocity"]).any())
        self.assertIn("anchor", components)


if __name__ == "__main__":
    unittest.main()
