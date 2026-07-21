import unittest

import torch

from gam_reg.models.gam_reg import GAMReg


def tiny_config():
    return {
        "model": {
            "in_channels": 1,
            "encoder_channels": [4, 8, 12, 16],
            "decoder_channels": [24, 16, 12, 8],
            "token_dim": 16,
            "use_anatomy_head": False,
            "num_anatomy_classes": 0,
            "tokenizers": {
                "coarse": {"feature_level": 3, "token_grid": [2, 2, 2]},
                "middle": {"feature_level": 2, "token_grid": [2, 2, 2]},
            },
            "matching": {
                "lambda_center": 1.0,
                "lambda_covariance": 0.5,
                "lambda_feature": 1.0,
                "lambda_anatomy": 0.0,
                "sinkhorn_epsilon": 0.1,
                "sinkhorn_iterations": 20,
                "middle_spatial_radius": 2.0,
            },
            "propagation": {"token_chunk": 4, "mahalanobis_clip": 30.0},
            "integration": {"steps": 3},
        }
    }


class ModelForwardTest(unittest.TestCase):
    def test_forward_shapes_and_zero_initialized_velocity(self):
        torch.manual_seed(5)
        model = GAMReg(tiny_config())
        moving = torch.randn(1, 1, 16, 16, 16)
        fixed = torch.randn(1, 1, 16, 16, 16)
        out = model(moving, fixed, return_debug=True)
        self.assertEqual(tuple(out["warped_moving"].shape), tuple(moving.shape))
        self.assertEqual(tuple(out["velocity"].shape), (1, 3, 16, 16, 16))
        self.assertEqual(tuple(out["phi_fwd"].shape), (1, 16, 16, 16, 3))
        self.assertEqual(tuple(out["gaussian_priors"]["U3"].shape[1:]), (3, 2, 2, 2))
        self.assertEqual(tuple(out["gaussian_priors"]["U2"].shape[1:]), (3, 4, 4, 4))
        self.assertIn("coarse", out["tokens_moving"])
        self.assertIn("middle", out["matches"])
        self.assertTrue(torch.allclose(out["velocity"], torch.zeros_like(out["velocity"])))

    def test_baseline_variant_forward_shapes(self):
        cfg = tiny_config()
        cfg["model"]["ablation_variant"] = "baseline_unet_registration"
        model = GAMReg(cfg)
        moving = torch.randn(1, 1, 16, 16, 16)
        fixed = torch.randn(1, 1, 16, 16, 16)
        out = model(moving, fixed)
        self.assertEqual(out["tokens_moving"], {})
        self.assertEqual(out["matches"], {})
        self.assertEqual(tuple(out["velocity"].shape), (1, 3, 16, 16, 16))


if __name__ == "__main__":
    unittest.main()
