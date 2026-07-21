import unittest

import torch

from gam_reg.models.gaussian_propagation import GaussianToVolumePropagator
from gam_reg.models.gaussian_types import GaussianMatchOutput, GaussianTokenBatch


class PropagationTest(unittest.TestCase):
    def test_single_anisotropic_token(self):
        mu = torch.zeros(1, 1, 3)
        sigma = torch.tensor([[[0.55, 0.15, 0.15]]], dtype=torch.float32)
        rotation = torch.eye(3).view(1, 1, 3, 3)
        cov = rotation @ torch.diag_embed(sigma.square()) @ rotation.transpose(-1, -2) + 1.0e-5 * torch.eye(3)
        tokens = GaussianTokenBatch(
            mu=mu,
            sigma=sigma,
            rotation=rotation,
            cov=cov,
            feat=torch.ones(1, 1, 4),
            anat_logits=torch.zeros(1, 1, 0),
            offset=torch.zeros(1, 1, 3),
        )
        disp = torch.tensor([[[0.25, -0.10, 0.05]]], dtype=torch.float32)
        match = GaussianMatchOutput(
            transport=torch.ones(1, 1, 1),
            row_prob=torch.ones(1, 1, 1),
            target_mu=mu + disp,
            displacement=disp,
            confidence=torch.ones(1, 1, 1),
            cost=torch.zeros(1, 1, 1),
        )
        prior, conf = GaussianToVolumePropagator(token_chunk=1)(tokens, match, (9, 9, 9))
        center = (0, slice(None), 4, 4, 4)
        self.assertTrue(torch.allclose(prior[center], disp[0, 0], atol=1e-4))
        self.assertGreater(float(conf[0, 0, 4, 4, 4]), float(conf[0, 0, 0, 0, 0]))
        self.assertGreater(float(conf[0, 0, 4, 4, 6]), float(conf[0, 0, 4, 6, 4]))


if __name__ == "__main__":
    unittest.main()
