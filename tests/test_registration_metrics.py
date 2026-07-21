import unittest

import torch

from gam_reg.metrics.registration_metrics import (
    available_hard_dice_per_class,
    registration_dice_metric_dict,
)
from gam_reg.models.spatial_transformer import identity_grid


class RegistrationDiceMetricsTest(unittest.TestCase):
    def test_identical_shared_class_has_unit_dice_and_zero_gain(self):
        moving = torch.zeros(1, 3, 8, 8, 8)
        fixed = torch.zeros_like(moving)
        moving[:, 0] = 1.0
        fixed[:, 0] = 1.0
        moving[:, 0, 2:5, 2:5, 2:5] = 0.0
        fixed[:, 0, 2:5, 2:5, 2:5] = 0.0
        moving[:, 1, 2:5, 2:5, 2:5] = 1.0
        fixed[:, 1, 2:5, 2:5, 2:5] = 1.0

        metrics = registration_dice_metric_dict(
            moving,
            fixed,
            identity_grid((8, 8, 8)),
        )
        self.assertAlmostEqual(float(metrics["dice_score_before"]), 1.0)
        self.assertAlmostEqual(float(metrics["dice_score_after"]), 1.0)
        self.assertAlmostEqual(float(metrics["dice_score_gain"]), 0.0)
        self.assertNotIn("dice_score_class_2_after", metrics)

    def test_one_sided_and_both_absent_classes_are_unavailable(self):
        moving = torch.zeros(1, 3, 8, 8, 8)
        fixed = torch.zeros_like(moving)
        moving[:, 0] = 1.0
        fixed[:, 0] = 1.0
        moving[:, 0, 2:4, 2:4, 2:4] = 0.0
        moving[:, 1, 2:4, 2:4, 2:4] = 1.0

        _, available = available_hard_dice_per_class(moving, fixed)
        self.assertFalse(bool(available.any()))
        metrics = registration_dice_metric_dict(
            moving,
            fixed,
            identity_grid((8, 8, 8)),
        )
        self.assertEqual(metrics, {})

    def test_misaligned_shared_class_has_subunit_hard_dice(self):
        moving = torch.zeros(1, 2, 8, 8, 8)
        fixed = torch.zeros_like(moving)
        moving[:, 0] = 1.0
        fixed[:, 0] = 1.0
        moving[:, 0, 1:3, 1:3, 1:3] = 0.0
        moving[:, 1, 1:3, 1:3, 1:3] = 1.0
        fixed[:, 0, 5:7, 5:7, 5:7] = 0.0
        fixed[:, 1, 5:7, 5:7, 5:7] = 1.0

        dice, available = available_hard_dice_per_class(moving, fixed)
        self.assertTrue(bool(available.all()))
        self.assertLess(float(dice[available].mean()), 0.01)


if __name__ == "__main__":
    unittest.main()
