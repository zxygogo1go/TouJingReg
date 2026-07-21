from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from gam_reg.config import load_config
from gam_reg.data.dataset import load_volume
from gam_reg.models.gam_reg import GAMReg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GAM-Reg inference for one moving/fixed pair")
    parser.add_argument("--config", default="gam_reg/configs/gam_reg_stable.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--moving", required=True)
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--output-dir", default="outputs/infer")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-npy", action="store_true")
    parser.add_argument("--target-shape", nargs=3, type=int)
    parser.add_argument(
        "--image-normalization",
        choices=["hu", "zero_one", "minus_one_one", "none"],
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    image_normalization = args.image_normalization or data_cfg.get("image_normalization", "hu")
    target_shape = args.target_shape or data_cfg.get("target_shape")
    device = torch.device(args.device)
    model = GAMReg(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    moving = load_volume(
        args.moving,
        image_normalization=image_normalization,
        target_shape=target_shape,
    ).unsqueeze(0).to(device)
    fixed = load_volume(
        args.fixed,
        image_normalization=image_normalization,
        target_shape=target_shape,
    ).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(moving, fixed, return_debug=False)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "warped_moving": outputs["warped_moving"].cpu(),
        "velocity": outputs["velocity"].cpu(),
        "phi_fwd": outputs["phi_fwd"].cpu(),
        "phi_inv": outputs["phi_inv"].cpu(),
    }
    torch.save(payload, out_dir / "inference.pt")
    if args.save_npy:
        np.save(out_dir / "warped_moving.npy", payload["warped_moving"].numpy())
        np.save(out_dir / "velocity.npy", payload["velocity"].numpy())
    print("saved %s" % (out_dir / "inference.pt"))


if __name__ == "__main__":
    main()
