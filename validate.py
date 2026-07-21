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
from gam_reg.metrics.jacobian_metrics import folding_ratio, mean_abs_det_j_minus_one
from gam_reg.models.gam_reg import GAMReg


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
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    model = GAMReg(cfg).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
    loss_fn = TotalRegistrationLoss(cfg).to(device)
    if args.synthetic or args.manifest is None:
        dataset = SyntheticRegistrationDataset(num_samples=args.max_batches, spatial_shape=(32, 40, 32))
    else:
        dataset = VolumePairDataset(args.manifest, data_root=args.data_root, num_seg_classes=cfg["model"].get("num_anatomy_classes"))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model.eval()
    sums = {}
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
            )
            components["folding_ratio_metric"] = folding_ratio(outputs["phi_fwd"])
            components["mean_abs_det_j_minus_one"] = mean_abs_det_j_minus_one(outputs["phi_fwd"])
            components["mean_abs_velocity"] = outputs["velocity"].abs().mean()
            for key, value in components.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach().cpu())
            count += 1
            if count >= args.max_batches:
                break
    metrics = {k: v / max(count, 1) for k, v in sums.items()}
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
