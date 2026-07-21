import unittest

import torch

from gam_reg.amp import make_grad_scaler, nonfinite_gradient_names, require_finite


class AmpUtilityTest(unittest.TestCase):
    def test_disabled_scaler_is_available_on_supported_torch_versions(self):
        scaler = make_grad_scaler(enabled=False)
        self.assertFalse(scaler.is_enabled())
        self.assertEqual(float(scaler.get_scale()), 1.0)

    def test_require_finite_accepts_finite_tensor(self):
        require_finite("value", torch.tensor([1.0, -2.0]))

    def test_require_finite_rejects_nan_and_inf(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaisesRegex(FloatingPointError, "non-finite test value"):
                require_finite("test value", torch.tensor(value))

    def test_nonfinite_gradient_names_reports_only_bad_parameters(self):
        good = torch.nn.Parameter(torch.tensor([1.0]))
        bad = torch.nn.Parameter(torch.tensor([2.0]))
        missing = torch.nn.Parameter(torch.tensor([3.0]))
        good.grad = torch.tensor([0.5])
        bad.grad = torch.tensor([float("inf")])
        names = nonfinite_gradient_names(
            [("good", good), ("bad", bad), ("missing", missing)]
        )
        self.assertEqual(names, ["bad"])


if __name__ == "__main__":
    unittest.main()
