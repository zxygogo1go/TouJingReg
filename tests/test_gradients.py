import unittest

import torch

from gam_reg.losses.total_loss import TotalRegistrationLoss
from gam_reg.models.gam_reg import GAMReg
from tests.test_model_forward import tiny_config


def grad_sum(param):
    if param.grad is None:
        return 0.0
    return float(param.grad.detach().abs().sum())


class EndToEndGradientTest(unittest.TestCase):
    def test_total_loss_backpropagates_to_required_modules(self):
        torch.manual_seed(11)
        cfg = tiny_config()
        model = GAMReg(cfg)
        loss_fn = TotalRegistrationLoss(cfg)
        moving = torch.randn(1, 1, 16, 16, 16)
        fixed = torch.randn(1, 1, 16, 16, 16)
        out = model(moving, fixed, return_debug=True)
        total, components = loss_fn(out, fixed=fixed)
        self.assertTrue(torch.isfinite(total))
        total.backward()

        self.assertGreater(grad_sum(model.tokenizer_coarse.center_head.weight), 0.0)
        self.assertGreater(grad_sum(model.tokenizer_coarse.scale_head.weight), 0.0)
        self.assertGreater(grad_sum(model.tokenizer_coarse.rotation_head.weight), 0.0)
        self.assertGreater(grad_sum(model.tokenizer_coarse.feature_projection.weight), 0.0)
        self.assertGreater(grad_sum(model.encoder.level0.block[0].weight), 0.0)
        self.assertGreater(grad_sum(model.decoder.velocity_head.weight), 0.0)
        for value in components.values():
            self.assertTrue(torch.isfinite(value.detach()).all())


if __name__ == "__main__":
    unittest.main()
