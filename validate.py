from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from gam_reg.config import load_config
from gam_reg.data.dataset import SyntheticRegistrationDataset, VolumePairDataset
from gam_reg.losses.total_loss import TotalRegistrationLoss
from gam_reg.metrics.jacobian_metrics import jacobian_metric_dict
from gam_reg.models.gam_reg import GAMReg
from gam_reg.training_schedule import stage_loss_weights


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate GAM-Reg")
    parser.add_argument("--config", default="gam_reg/configs/gam_reg_stable.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--manifest")
    parser.add_argument("--data-root")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output-json")
    parser.add_argument("--target-shape", nargs=3, type=int)
    parser.add_argument(
        "--image-normalization",
        choices=["hu", "zero_one", "minus_one_one", "none"],
    )
    args = parser.parse_args()
    if args.batch_size != 1:
        parser.error("per-pair validation currently requires --batch-size 1")

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    image_normalization = args.image_normalization or data_cfg.get("image_normalization", "hu")
    target_shape = args.target_shape or data_cfg.get("target_shape")
    device = torch.device(args.device)
    model = GAMReg(cfg).to(device)
    checkpoint_stage = "joint"
    checkpoint_stage_step = 0
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        checkpoint_stage = ckpt.get("stage", "joint")
        checkpoint_stage_step = int(ckpt.get("stage_step", ckpt.get("step", 0)))
    loss_fn = TotalRegistrationLoss(cfg).to(device)
    effective_weights = stage_loss_weights(
        loss_fn.weights,
        stage=checkpoint_stage,
        stage_step=checkpoint_stage_step,
        training_config=cfg["training"],
    )
    if args.synthetic or args.manifest is None:
        dataset = SyntheticRegistrationDataset(num_samples=args.max_batches, spatial_shape=(32, 40, 32))
    else:
        dataset = VolumePairDataset(
            args.manifest,
            data_root=args.data_root,
            num_seg_classes=cfg["model"].get("num_anatomy_classes"),
            image_normalization=image_normalization,
            target_shape=target_shape,
            spacing_dhw=data_cfg.get("spacing_dhw", [1.0, 1.0, 1.0]),
        )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model.eval()
    sums = {}
    pair_results = []
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            outputs = model(batch["moving"], batch["fixed"], batch.get("moving_seg"), batch.get("fixed_seg"), return_debug=True)
            _, components = loss_fn(
                outputs,
                fixed=batch["fixed"],
                moving=batch["moving"],
                moving_seg=batch.get("moving_seg"),
                fixed_seg=batch.get("fixed_seg"),
                spacing_dhw=batch.get("spacing_dhw"),
                weights_override=effective_weights,
            )
            components.update(
                jacobian_metric_dict(
                    outputs["phi_fwd"],
                    minimum_determinant=loss_fn.jacobian_minimum_determinant,
                    phi_inv=outputs["phi_inv"],
                )
            )
            components["mean_abs_velocity"] = outputs["velocity"].abs().mean()
            pair_metrics = {key: float(value.detach().cpu()) for key, value in components.items()}
            for key, value in components.items():
                sums[key] = sums.get(key, 0.0) + pair_metrics[key]
            if hasattr(dataset, "rows"):
                row = dataset.rows[count]
                moving_name = row["moving"]
                fixed_name = row["fixed"]
            else:
                moving_name = "synthetic_%04d_moving" % count
                fixed_name = "synthetic_%04d_fixed" % count
            pair_results.append(
                {
                    "index": count,
                    "moving": moving_name,
                    "fixed": fixed_name,
                    "metrics": pair_metrics,
                }
            )
            count += 1
            del outputs, components, batch, value, _
            if count >= args.max_batches:
                break
    metrics = {k: v / max(count, 1) for k, v in sums.items()}
    metrics["mean_pair_minimum_det_j"] = metrics.pop("minimum_det_j", 1.0)
    metrics["max_pair_folding_ratio_metric"] = max(
        (pair["metrics"]["folding_ratio_metric"] for pair in pair_results),
        default=0.0,
    )
    metrics["max_pair_below_minimum_det_j_ratio"] = max(
        (pair["metrics"]["below_minimum_det_j_ratio"] for pair in pair_results),
        default=0.0,
    )
    metrics["minimum_pair_det_j"] = min(
        (pair["metrics"]["minimum_det_j"] for pair in pair_results),
        default=1.0,
    )
    report = {
        "summary": metrics,
        "num_pairs": count,
        "checkpoint_stage": checkpoint_stage,
        "checkpoint_stage_step": checkpoint_stage_step,
        "effective_weights": effective_weights,
        "pairs": pair_results,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
