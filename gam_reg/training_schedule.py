from __future__ import annotations

from typing import Any, Dict, Mapping


def stage_loss_weights(
    base_weights: Mapping[str, float],
    stage: str,
    stage_step: int,
    training_config: Mapping[str, Any],
) -> Dict[str, float]:
    weights = {key: float(value) for key, value in base_weights.items()}
    if stage != "registration-warmup":
        return weights

    schedule = training_config.get("stage_schedules", {}).get("registration-warmup", {})
    ramp_steps = int(schedule.get("ramp_steps", 2000))
    if ramp_steps < 0:
        raise ValueError("registration-warmup ramp_steps must be non-negative")
    if stage_step < 0:
        raise ValueError("stage_step must be non-negative")
    progress = 1.0 if ramp_steps == 0 else min(float(stage_step) / float(ramp_steps), 1.0)
    for key in ("anchor", "jacobian"):
        target = weights.get(key, 0.0)
        start = float(schedule.get(key + "_start", 0.0))
        weights[key] = start + progress * (target - start)
    return weights
