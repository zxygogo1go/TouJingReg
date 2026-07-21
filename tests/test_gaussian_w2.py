import unittest

import torch

from gam_reg.models.gaussian_tokenizer import rotation_6d_to_matrix
from gam_reg.models.gaussian_wasserstein import pairwise_gaussian_w2


def build_cov(sigma, rot6):
    rotation = rotation_6d_to_matrix(rot6)
    return rotation @ torch.diag_embed(sigma.square()) @ rotation.transpose(-1, -2) + 1.0e-5 * torch.eye(3)


class GaussianW2Test(unittest.TestCase):
    def test_identity_symmetry_and_nonnegative(self):
        mu1 = torch.tensor([[[0.1, -0.2, 0.3]]], dtype=torch.float32)
        sigma1 = torch.tensor([[[0.2, 0.3, 0.4]]], dtype=torch.float32)
        rot6 = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]], dtype=torch.float32)
        cov1 = build_cov(sigma1, rot6)
        _, _, w_same = pairwise_gaussian_w2(mu1, cov1, mu1, cov1)
        self.assertTrue(torch.allclose(w_same, torch.zeros_like(w_same), atol=2e-5))

        mu2 = torch.tensor([[[-0.2, 0.4, -0.1], [0.5, 0.1, 0.2]]], dtype=torch.float32)
        sigma2 = torch.tensor([[[0.5, 0.4, 0.3], [0.2, 0.6, 0.4]]], dtype=torch.float32)
        cov2 = build_cov(sigma2, rot6.expand(1, 2, 6))
        _, _, w12 = pairwise_gaussian_w2(mu1, cov1, mu2, cov2)
        _, _, w21 = pairwise_gaussian_w2(mu2, cov2, mu1, cov1)
        self.assertTrue(torch.all(w12 >= 0))
        self.assertTrue(torch.allclose(w12, w21.transpose(1, 2), atol=1e-5))

    def test_gradients_flow_to_mu_sigma_and_rotation(self):
        torch.manual_seed(3)
        mu1 = torch.randn(1, 2, 3, requires_grad=True) * 0.1
        mu1.retain_grad()
        mu2 = torch.randn(1, 3, 3) * 0.1
        sigma1 = (torch.rand(1, 2, 3, requires_grad=True) + 0.2)
        sigma1.retain_grad()
        sigma2 = torch.rand(1, 3, 3) + 0.2
        rot6 = torch.randn(1, 2, 6, requires_grad=True)
        rot6.retain_grad()
        fixed_rot6 = torch.randn(1, 3, 6)
        cov1 = build_cov(sigma1, rot6)
        cov2 = build_cov(sigma2, fixed_rot6)
        _, cov_cost, w2 = pairwise_gaussian_w2(mu1, cov1, mu2, cov2)
        loss = w2.mean() + cov_cost.mean()
        loss.backward()
        self.assertTrue(torch.isfinite(mu1.grad).all())
        self.assertTrue(torch.isfinite(sigma1.grad).all())
        self.assertTrue(torch.isfinite(rot6.grad).all())
        self.assertGreater(float(mu1.grad.abs().sum()), 0.0)
        self.assertGreater(float(sigma1.grad.abs().sum()), 0.0)
        self.assertGreater(float(rot6.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
