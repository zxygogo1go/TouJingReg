import unittest

import torch

from gam_reg.models.encoder import SharedRegistrationEncoder
from gam_reg.models.gaussian_tokenizer import GaussianAnatomyTokenizer


class EncoderTokenizerTest(unittest.TestCase):
    def test_encoder_shapes(self):
        encoder = SharedRegistrationEncoder(in_channels=1, channels=(4, 8, 12, 16))
        features = encoder(torch.randn(1, 1, 16, 20, 16))
        self.assertEqual([tuple(f.shape) for f in features], [
            (1, 4, 16, 20, 16),
            (1, 8, 8, 10, 8),
            (1, 12, 4, 5, 4),
            (1, 16, 2, 3, 2),
        ])

    def test_tokenizer_outputs_valid_gaussians_and_gradients(self):
        torch.manual_seed(7)
        tokenizer = GaussianAnatomyTokenizer(
            in_channels=12,
            token_dim=16,
            token_grid=(2, 2, 2),
            use_anatomy_head=True,
            num_anatomy_classes=5,
        )
        feature = torch.randn(1, 12, 5, 6, 5, requires_grad=True)
        tokens = tokenizer(feature)
        self.assertEqual(tuple(tokens.mu.shape), (1, 8, 3))
        self.assertEqual(tuple(tokens.cov.shape), (1, 8, 3, 3))
        self.assertEqual(tuple(tokens.feat.shape), (1, 8, 16))
        self.assertEqual(tuple(tokens.anat_logits.shape), (1, 8, 5))
        self.assertLessEqual(float(tokens.mu.abs().max()), 1.0)
        self.assertTrue(torch.all(tokens.sigma > 0))
        eig = torch.linalg.eigvalsh(tokens.cov.float())
        self.assertTrue(torch.all(eig > 0))
        feat_norm = tokens.feat.norm(dim=-1)
        self.assertTrue(torch.allclose(feat_norm, torch.ones_like(feat_norm), atol=1e-5))
        loss = tokens.mu.square().sum() + tokens.sigma.sum() + tokens.rotation.sum() + tokens.feat.sum()
        loss.backward()
        self.assertTrue(torch.isfinite(tokenizer.feature_projection.weight.grad).all())
        self.assertGreater(float(tokenizer.feature_projection.weight.grad.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(tokenizer.center_head.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
