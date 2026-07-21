from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gam_reg.config import load_config
from gam_reg.data.dataset import SyntheticRegistrationDataset, VolumePairDataset
from gam_reg.losses.total_loss import TotalRegistrationLoss
from gam_reg.models.gam_reg import ABLATION_VARIANTS, GAMReg


ORDERED_VARIANTS = [
    "baseline_unet_registration",
    "point_tokens",
    "isotropic_gaussian",
    "anisotropic_gaussian_without_w2",
    "anisotropic_gaussian_w2_no_sinkhorn",
    "full",
    "full_without_anchor_loss",
    "full_without_dice",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GAM-Reg ablation variants")
    parser.add_argument("--config", default="gam_reg/configs/gam_reg_stable.yaml")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--manifest")
    parser.add_argument("--data-root")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target-shape", nargs=3, type=int)
    parser.add_argument(
        "--image-normalization",
        choices=["hu", "zero_one", "minus_one_one", "none"],
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    data_cfg = base_cfg.get("data", {})
    image_normalization = args.image_normalization or data_cfg.get("image_normalization", "hu")
    target_shape = args.target_shape or data_cfg.get("target_shape")
    if args.synthetic or args.manifest is None:
        dataset = SyntheticRegistrationDataset(num_samples=args.max_batches, spatial_shape=(32, 40, 32))
    else:
        dataset = VolumePairDataset(
            args.manifest,
            data_root=args.data_root,
            num_seg_classes=base_cfg["model"].get("num_anatomy_classes"),
            image_normalization=image_normalization,
            target_shape=target_shape,
        )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    results = {}
    for variant in ORDERED_VARIANTS:
        if variant not in ABLATION_VARIANTS:
            continue
        cfg = deepcopy(base_cfg)
        cfg["model"]["ablation_variant"] = variant
        model = GAMReg(cfg).to(device)
        if args.checkpoint_dir:
            ckpt_path = Path(args.checkpoint_dir) / ("%s.pt" % variant)
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(ckpt["model"])
        loss_fn = TotalRegistrationLoss(cfg).to(device)
        sums = {}
        count = 0
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(batch["moving"], batch["fixed"], batch.get("moving_seg"), batch.get("fixed_seg"), return_debug=True)
                _, components = loss_fn(outputs, fixed=batch["fixed"], moving=batch["moving"], moving_seg=batch.get("moving_seg"), fixed_seg=batch.get("fixed_seg"))
                components["mean_abs_velocity"] = outputs["velocity"].abs().mean()
                for key, value in components.items():
                    sums[key] = sums.get(key, 0.0) + float(value.detach().cpu())
                count += 1
                if count >= args.max_batches:
                    break
        results[variant] = {k: v / max(count, 1) for k, v in sums.items()}
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
