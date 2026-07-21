from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from gam_reg.config import default_config, deep_update
from gam_reg.losses.deformation_losses import jacobian_folding_penalty, smoothness_loss
from gam_reg.losses.dice import dice_loss
from gam_reg.losses.feature_similarity import multi_scale_feature_similarity_loss
from gam_reg.losses.gaussian_losses import (
    anatomy_token_loss,
    gaussian_anchor_consistency_loss,
    token_regularization_loss,
)
from gam_reg.losses.lncc import LNCCLoss


class TotalRegistrationLoss(nn.Module):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        cfg = default_config()
        if config is not None:
            cfg = deep_update(cfg, config)
        self.config = cfg
        self.weights = dict(cfg["loss"]["weights"])
        self.jacobian_minimum_determinant = float(
            cfg["loss"].get("jacobian_minimum_determinant", 0.05)
        )
        self.jacobian_tail_fraction = float(cfg["loss"].get("jacobian_tail_fraction", 0.001))
        self.jacobian_tail_weight = float(cfg["loss"].get("jacobian_tail_weight", 0.25))
        variant = cfg.get("model", {}).get("ablation_variant", "full")
        if variant == "full_without_anchor_loss":
            self.weights["anchor"] = 0.0
        if variant == "full_without_dice":
            self.weights["dice"] = 0.0
        self.lncc = LNCCLoss(cfg["loss"].get("lncc_window", [9, 9, 9]))

    def forward(
        self,
        outputs: Dict[str, Any],
        fixed: torch.Tensor,
        moving: Optional[torch.Tensor] = None,
        moving_seg: Optional[torch.Tensor] = None,
        fixed_seg: Optional[torch.Tensor] = None,
        spacing_dhw: Optional[torch.Tensor] = None,
        weights_override: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        warped = outputs["warped_moving"]
        phi_fwd = outputs["phi_fwd"]
        phi_inv = outputs["phi_inv"]
        velocity = outputs["velocity"]
        components: Dict[str, torch.Tensor] = {}

        components["sim"] = self.lncc(fixed, warped)
        debug = outputs.get("debug", {})
        if "features_moving" in debug and "features_fixed" in debug:
            components["feature"] = multi_scale_feature_similarity_loss(
                debug["features_moving"],
                debug["features_fixed"],
                phi_inv,
            )
        else:
            components["feature"] = velocity.sum() * 0.0
        components["smooth"] = smoothness_loss(velocity, spacing_dhw=spacing_dhw)
        components["jacobian"], components["folding_ratio"] = jacobian_folding_penalty(
            phi_fwd,
            phi_inv=phi_inv,
            minimum_determinant=self.jacobian_minimum_determinant,
            tail_fraction=self.jacobian_tail_fraction,
            tail_weight=self.jacobian_tail_weight,
        )
        components["dice"] = dice_loss(moving_seg, fixed_seg, phi_inv)
        components["anchor"] = gaussian_anchor_consistency_loss(
            outputs.get("tokens_moving", {}),
            outputs.get("matches", {}),
            phi_fwd,
        )
        components["anatomy"] = anatomy_token_loss(
            outputs.get("tokens_moving", {}),
            outputs.get("tokens_fixed", {}),
            moving_seg,
            fixed_seg,
        ).to(velocity.device)
        token_reg = token_regularization_loss(outputs.get("tokens_moving", {})).to(velocity.device)
        token_reg = token_reg + token_regularization_loss(outputs.get("tokens_fixed", {})).to(velocity.device)
        components["token_regularization"] = token_reg

        effective_weights = dict(self.weights)
        if weights_override is not None:
            effective_weights.update({key: float(value) for key, value in weights_override.items()})
        total = velocity.sum() * 0.0
        for key, weight in effective_weights.items():
            total = total + float(weight) * components[key]
        components["total"] = total
        return total, components
