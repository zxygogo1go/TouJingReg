from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def default_config() -> Dict[str, Any]:
    return {
        "model": {
            "name": "GAMReg",
            "in_channels": 1,
            "encoder_channels": [16, 32, 64, 128],
            "decoder_channels": [256, 128, 64, 32],
            "token_dim": 96,
            "use_anatomy_head": True,
            "num_anatomy_classes": 5,
            "ablation_variant": "full",
            "tokenizers": {
                "coarse": {
                    "feature_level": 3,
                    "token_grid": [4, 4, 4],
                    "sigma_min_ratio": 0.20,
                    "sigma_max_ratio": 1.20,
                    "offset_ratio": 0.35,
                },
                "middle": {
                    "feature_level": 2,
                    "token_grid": [6, 6, 6],
                    "sigma_min_ratio": 0.20,
                    "sigma_max_ratio": 1.20,
                    "offset_ratio": 0.35,
                },
            },
            "matching": {
                "lambda_center": 1.0,
                "lambda_covariance": 0.5,
                "lambda_feature": 1.0,
                "lambda_anatomy": 0.2,
                "sinkhorn_epsilon": 0.07,
                "sinkhorn_iterations": 30,
                "middle_spatial_radius": 1.0,
                "large_cost": 10000.0,
            },
            "propagation": {
                "token_chunk": 32,
                "mahalanobis_clip": 30.0,
            },
            "integration": {
                "steps": 7,
            },
        },
        "loss": {
            "lncc_window": [9, 9, 9],
            "jacobian_minimum_determinant": 0.05,
            "jacobian_tail_fraction": 0.001,
            "jacobian_tail_weight": 0.25,
            "weights": {
                "sim": 1.0,
                "feature": 0.20,
                "smooth": 0.10,
                "jacobian": 5.0,
                "dice": 1.0,
                "anchor": 0.50,
                "anatomy": 0.10,
                "token_regularization": 0.01,
            },
        },
        "training": {
            "optimizer": "adamw",
            "learning_rate": 0.0001,
            "weight_decay": 0.00001,
            "gradient_clip_norm": 1.0,
            "amp": True,
            "amp_init_scale": 1024.0,
            "amp_growth_interval": 2000,
            "amp_max_retries": 8,
            "batch_size": 1,
            "stage_schedules": {
                "registration-warmup": {
                    "ramp_steps": 2000,
                    "anchor_start": 0.1,
                    "jacobian_start": 0.5,
                }
            },
        },
        "data": {
            "image_normalization": "hu",
            "target_shape": None,
            "spacing_dhw": [1.0, 1.0, 1.0],
        },
    }


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = default_config()
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        cfg = deep_update(cfg, loaded)
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg
