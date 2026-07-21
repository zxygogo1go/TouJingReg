from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import nn

from gam_reg.config import default_config, deep_update
from gam_reg.models.diffeomorphic import DiffeomorphicIntegrator
from gam_reg.models.encoder import SharedRegistrationEncoder
from gam_reg.models.gaussian_matcher import LogSinkhornMatcher
from gam_reg.models.gaussian_propagation import GaussianToVolumePropagator
from gam_reg.models.gaussian_tokenizer import GaussianAnatomyTokenizer
from gam_reg.models.gaussian_types import GaussianTokenBatch
from gam_reg.models.spatial_transformer import spatial_transform
from gam_reg.models.velocity_decoder import ResidualVelocityDecoder


ABLATION_VARIANTS = {
    "full",
    "baseline_unet_registration",
    "point_tokens",
    "isotropic_gaussian",
    "anisotropic_gaussian_without_w2",
    "anisotropic_gaussian_w2_no_sinkhorn",
    "full_without_anchor_loss",
    "full_without_dice",
}


def _as_model_config(config: Dict[str, Any] | None) -> Dict[str, Any]:
    if config is None:
        return default_config()["model"]
    if "model" in config:
        return deep_update(default_config(), config)["model"]
    return deep_update(default_config()["model"], config)


def _geometry_adjusted_tokens(tokens: GaussianTokenBatch, mode: str, cov_eps: float = 1.0e-5) -> GaussianTokenBatch:
    if mode == "full":
        return tokens
    b, n, _ = tokens.mu.shape
    dtype = tokens.mu.dtype
    device = tokens.mu.device
    eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).expand(b, n, 3, 3)
    if mode == "isotropic":
        sigma = tokens.sigma.mean(dim=-1, keepdim=True).expand_as(tokens.sigma)
    elif mode == "point":
        sigma = tokens.sigma.mean(dim=-1, keepdim=True).expand_as(tokens.sigma) * 0.35
    else:
        raise ValueError("unknown geometry mode")
    cov = eye @ torch.diag_embed(sigma.square()) @ eye.transpose(-1, -2)
    cov = cov + cov_eps * torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3)
    return GaussianTokenBatch(
        mu=tokens.mu,
        sigma=sigma,
        rotation=eye,
        cov=cov,
        feat=tokens.feat,
        anat_logits=tokens.anat_logits,
        offset=tokens.offset,
    )


class GAMReg(nn.Module):
    """Gaussian Anatomy Matching Registration network."""

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__()
        cfg = _as_model_config(config)
        self.config = cfg
        self.variant = str(cfg.get("ablation_variant", "full"))
        if self.variant not in ABLATION_VARIANTS:
            raise ValueError("unknown ablation_variant: %s" % self.variant)

        encoder_channels = cfg.get("encoder_channels", [16, 32, 64, 128])
        decoder_channels = cfg.get("decoder_channels", [256, 128, 64, 32])
        token_dim = int(cfg.get("token_dim", 96))
        use_anatomy = bool(cfg.get("use_anatomy_head", False))
        num_anatomy = int(cfg.get("num_anatomy_classes", 0)) if use_anatomy else 0

        self.encoder = SharedRegistrationEncoder(
            in_channels=int(cfg.get("in_channels", 1)),
            channels=encoder_channels,
        )
        coarse_cfg = cfg["tokenizers"]["coarse"]
        middle_cfg = cfg["tokenizers"]["middle"]
        self.tokenizer_coarse = GaussianAnatomyTokenizer(
            in_channels=encoder_channels[int(coarse_cfg.get("feature_level", 3))],
            token_dim=token_dim,
            token_grid=coarse_cfg.get("token_grid", [4, 4, 4]),
            sigma_min_ratio=coarse_cfg.get("sigma_min_ratio", 0.20),
            sigma_max_ratio=coarse_cfg.get("sigma_max_ratio", 1.20),
            offset_ratio=coarse_cfg.get("offset_ratio", 0.35),
            use_anatomy_head=use_anatomy,
            num_anatomy_classes=num_anatomy,
        )
        self.tokenizer_middle = GaussianAnatomyTokenizer(
            in_channels=encoder_channels[int(middle_cfg.get("feature_level", 2))],
            token_dim=token_dim,
            token_grid=middle_cfg.get("token_grid", [6, 6, 6]),
            sigma_min_ratio=middle_cfg.get("sigma_min_ratio", 0.20),
            sigma_max_ratio=middle_cfg.get("sigma_max_ratio", 1.20),
            offset_ratio=middle_cfg.get("offset_ratio", 0.35),
            use_anatomy_head=use_anatomy,
            num_anatomy_classes=num_anatomy,
        )

        matching_cfg = dict(cfg["matching"])
        if self.variant in {"point_tokens", "anisotropic_gaussian_without_w2"}:
            matching_cfg["lambda_covariance"] = 0.0
        use_sinkhorn = self.variant != "anisotropic_gaussian_w2_no_sinkhorn"
        self.matcher_coarse = LogSinkhornMatcher(
            lambda_center=matching_cfg.get("lambda_center", 1.0),
            lambda_covariance=matching_cfg.get("lambda_covariance", 0.5),
            lambda_feature=matching_cfg.get("lambda_feature", 1.0),
            lambda_anatomy=matching_cfg.get("lambda_anatomy", 0.2),
            sinkhorn_epsilon=matching_cfg.get("sinkhorn_epsilon", 0.07),
            sinkhorn_iterations=matching_cfg.get("sinkhorn_iterations", 30),
            spatial_radius=None,
            large_cost=matching_cfg.get("large_cost", 10000.0),
            use_sinkhorn=use_sinkhorn,
        )
        self.matcher_middle = LogSinkhornMatcher(
            lambda_center=matching_cfg.get("lambda_center", 1.0),
            lambda_covariance=matching_cfg.get("lambda_covariance", 0.5),
            lambda_feature=matching_cfg.get("lambda_feature", 1.0),
            lambda_anatomy=matching_cfg.get("lambda_anatomy", 0.2),
            sinkhorn_epsilon=matching_cfg.get("sinkhorn_epsilon", 0.07),
            sinkhorn_iterations=matching_cfg.get("sinkhorn_iterations", 30),
            spatial_radius=matching_cfg.get("middle_spatial_radius", 1.0),
            large_cost=matching_cfg.get("large_cost", 10000.0),
            use_sinkhorn=use_sinkhorn,
        )
        prop_cfg = cfg["propagation"]
        self.propagator = GaussianToVolumePropagator(
            token_chunk=prop_cfg.get("token_chunk", 32),
            mahalanobis_clip=prop_cfg.get("mahalanobis_clip", 30.0),
        )
        self.decoder = ResidualVelocityDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
        )
        self.integrator = DiffeomorphicIntegrator(steps=cfg["integration"].get("steps", 7))

    def _variant_geometry_mode(self) -> str:
        if self.variant == "point_tokens":
            return "point"
        if self.variant == "isotropic_gaussian":
            return "isotropic"
        return "full"

    def forward(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
        moving_seg: torch.Tensor | None = None,
        fixed_seg: torch.Tensor | None = None,
        return_debug: bool = False,
    ) -> Dict[str, Any]:
        if moving.ndim != 5 or fixed.ndim != 5:
            raise AssertionError("moving/fixed must have shape [B,1,D,H,W]")
        if moving.shape != fixed.shape:
            raise AssertionError("moving and fixed must be resampled to identical shapes")

        fm = self.encoder(moving)
        ff = self.encoder(fixed)
        b = moving.shape[0]

        if self.variant == "baseline_unet_registration":
            u3 = moving.new_zeros((b, 3) + tuple(fm[3].shape[-3:]))
            c3 = moving.new_zeros((b, 1) + tuple(fm[3].shape[-3:]))
            u2 = moving.new_zeros((b, 3) + tuple(fm[2].shape[-3:]))
            c2 = moving.new_zeros((b, 1) + tuple(fm[2].shape[-3:]))
            tokens_moving: Dict[str, GaussianTokenBatch] = {}
            tokens_fixed: Dict[str, GaussianTokenBatch] = {}
            matches = {}
        else:
            tm3 = self.tokenizer_coarse(fm[3])
            tf3 = self.tokenizer_coarse(ff[3])
            tm2 = self.tokenizer_middle(fm[2])
            tf2 = self.tokenizer_middle(ff[2])
            mode = self._variant_geometry_mode()
            tm3 = _geometry_adjusted_tokens(tm3, mode)
            tf3 = _geometry_adjusted_tokens(tf3, mode)
            tm2 = _geometry_adjusted_tokens(tm2, mode)
            tf2 = _geometry_adjusted_tokens(tf2, mode)
            m3 = self.matcher_coarse(tm3, tf3)
            m2 = self.matcher_middle(tm2, tf2)
            u3, c3 = self.propagator(tm3, m3, fm[3].shape[-3:])
            u2, c2 = self.propagator(tm2, m2, fm[2].shape[-3:])
            tokens_moving = {"coarse": tm3, "middle": tm2}
            tokens_fixed = {"coarse": tf3, "middle": tf2}
            matches = {"coarse": m3, "middle": m2}

        velocity = self.decoder(fm, ff, u3, c3, u2, c2)
        phi_fwd, phi_inv = self.integrator(velocity)
        warped_moving = spatial_transform(moving, phi_inv)
        output: Dict[str, Any] = {
            "warped_moving": warped_moving,
            "velocity": velocity,
            "phi_fwd": phi_fwd,
            "phi_inv": phi_inv,
            "tokens_moving": tokens_moving,
            "tokens_fixed": tokens_fixed,
            "matches": matches,
            "gaussian_priors": {"U3": u3, "C3": c3, "U2": u2, "C2": c2},
        }
        if return_debug:
            output["debug"] = {
                "features_moving": fm,
                "features_fixed": ff,
                "moving_seg": moving_seg,
                "fixed_seg": fixed_seg,
            }
        return output
