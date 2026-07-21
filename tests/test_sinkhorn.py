import unittest

import torch

from gam_reg.models.gaussian_matcher import LogSinkhornMatcher
from gam_reg.models.gaussian_types import GaussianTokenBatch
from gam_reg.models.sinkhorn import log_sinkhorn


def make_tokens(mu):
    b, n, _ = mu.shape
    sigma = torch.full((b, n, 3), 0.25)
    rotation = torch.eye(3).view(1, 1, 3, 3).expand(b, n, 3, 3).clone()
    cov = rotation @ torch.diag_embed(sigma.square()) @ rotation.transpose(-1, -2) + 1.0e-5 * torch.eye(3)
    feat = torch.nn.functional.normalize(torch.randn(b, n, 8), dim=-1)
    anat = torch.zeros(b, n, 0)
    offset = torch.zeros(b, n, 3)
    return GaussianTokenBatch(mu=mu, sigma=sigma, rotation=rotation, cov=cov, feat=feat, anat_logits=anat, offset=offset)


class SinkhornTest(unittest.TestCase):
    def test_log_sinkhorn_balanced_marginals(self):
        torch.manual_seed(2)
        cost = torch.rand(2, 4, 5)
        p = log_sinkhorn(cost, epsilon=0.1, iterations=200, convergence_tol=1e-7)
        self.assertTrue(torch.isfinite(p).all())
        self.assertTrue(torch.all(p >= 0))
        self.assertTrue(torch.allclose(p.sum(dim=-1), torch.full((2, 4), 0.25), atol=1e-4))
        self.assertTrue(torch.allclose(p.sum(dim=-2), torch.full((2, 5), 0.20), atol=1e-4))

    def test_matcher_outputs_barycenters_and_confidence(self):
        moving = make_tokens(torch.tensor([[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]], dtype=torch.float32))
        fixed = make_tokens(torch.tensor([[[-0.4, 0.0, 0.0], [0.4, 0.0, 0.0]]], dtype=torch.float32))
        matcher = LogSinkhornMatcher(
            lambda_center=3.0,
            lambda_covariance=0.0,
            lambda_feature=0.0,
            lambda_anatomy=0.0,
            sinkhorn_epsilon=0.03,
            sinkhorn_iterations=100,
        )
        out = matcher(moving, fixed)
        self.assertEqual(tuple(out.transport.shape), (1, 2, 2))
        self.assertTrue(torch.isfinite(out.transport).all())
        self.assertTrue(torch.all(out.confidence >= 0))
        self.assertTrue(torch.all(out.confidence <= 1))
        self.assertLess(float(out.target_mu[0, 0, 0]), 0.0)
        self.assertGreater(float(out.target_mu[0, 1, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
