import unittest

from gam_reg.training_schedule import stage_loss_weights


class TrainingScheduleTest(unittest.TestCase):
    def setUp(self):
        self.base = {"sim": 1.0, "anchor": 0.5, "jacobian": 0.1}
        self.training = {
            "stage_schedules": {
                "registration-warmup": {
                    "ramp_steps": 100,
                    "anchor_start": 0.1,
                    "jacobian_start": 0.0,
                }
            }
        }

    def test_registration_warmup_ramps_anchor_and_jacobian(self):
        start = stage_loss_weights(self.base, "registration-warmup", 0, self.training)
        middle = stage_loss_weights(self.base, "registration-warmup", 50, self.training)
        end = stage_loss_weights(self.base, "registration-warmup", 100, self.training)
        self.assertAlmostEqual(start["anchor"], 0.1)
        self.assertAlmostEqual(start["jacobian"], 0.0)
        self.assertAlmostEqual(middle["anchor"], 0.3)
        self.assertAlmostEqual(middle["jacobian"], 0.05)
        self.assertAlmostEqual(end["anchor"], 0.5)
        self.assertAlmostEqual(end["jacobian"], 0.1)

    def test_joint_uses_base_weights(self):
        actual = stage_loss_weights(self.base, "joint", 0, self.training)
        self.assertEqual(actual, self.base)

    def test_invalid_schedule_is_rejected(self):
        training = {"stage_schedules": {"registration-warmup": {"ramp_steps": -1}}}
        with self.assertRaisesRegex(ValueError, "ramp_steps"):
            stage_loss_weights(self.base, "registration-warmup", 0, training)


if __name__ == "__main__":
    unittest.main()
