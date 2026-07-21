from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from gam_reg.config import load_config
from gam_reg.data.dataset import SyntheticRegistrationDataset, VolumePairDataset
from gam_reg.losses.total_loss import TotalRegistrationLoss
from gam_reg.models.gam_reg import GAMReg


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def build_dataset(args, cfg):
    if args.stage == "synthetic-warmup" or args.manifest is None:
        return SyntheticRegistrationDataset(
            num_samples=max(args.steps_per_epoch * cfg["training"]["batch_size"], 1),
            spatial_shape=args.synthetic_shape,
            seed=args.seed,
        )
    return VolumePairDataset(
        args.manifest,
        data_root=args.data_root,
        num_seg_classes=cfg["model"].get("num_anatomy_classes"),
        image_normalization=args.image_normalization,
        target_shape=args.target_shape,
    )


def save_checkpoint(path: Path, model, optimizer, epoch: int, step: int, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": cfg,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GAM-Reg")
    parser.add_argument("--config", default="gam_reg/configs/gam_reg_stable.yaml")
    parser.add_argument("--stage", choices=["synthetic-warmup", "registration-warmup", "joint"], default="synthetic-warmup")
    parser.add_argument("--manifest")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir", default="runs/gam_reg")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--synthetic-shape", nargs=3, type=int, default=[32, 40, 32])
    parser.add_argument("--target-shape", nargs=3, type=int)
    parser.add_argument(
        "--image-normalization",
        choices=["hu", "zero_one", "minus_one_one", "none"],
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    if args.image_normalization is None:
        args.image_normalization = data_cfg.get("image_normalization", "hu")
    if args.target_shape is None:
        args.target_shape = data_cfg.get("target_shape")
    device = torch.device(args.device)
    model = GAMReg(cfg).to(device)
    loss_fn = TotalRegistrationLoss(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("step", 0))

    dataset = build_dataset(args, cfg)
    loader = DataLoader(dataset, batch_size=int(cfg["training"]["batch_size"]), shuffle=True, num_workers=0)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(args.output_dir)
    except Exception:
        writer = None

    use_amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    model.train()
    output_dir = Path(args.output_dir)
    for epoch in range(start_epoch, int(args.epochs)):
        running = {}
        loader_iter = iter(loader)
        for step in range(int(args.steps_per_epoch)):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(batch["moving"], batch["fixed"], batch.get("moving_seg"), batch.get("fixed_seg"), return_debug=True)
                total, components = loss_fn(
                    outputs,
                    fixed=batch["fixed"],
                    moving=batch["moving"],
                    moving_seg=batch.get("moving_seg"),
                    fixed_seg=batch.get("fixed_seg"),
                )
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            for key, value in components.items():
                scalar = float(value.detach().cpu())
                running[key] = running.get(key, 0.0) + scalar
                if writer is not None:
                    writer.add_scalar("train/" + key, scalar, global_step)
            if step % 10 == 0:
                print(json.dumps({"epoch": epoch, "step": step, "loss": float(total.detach().cpu())}))
        means = {k: v / float(args.steps_per_epoch) for k, v in running.items()}
        print(json.dumps({"epoch": epoch, "mean": means}, indent=2))
        save_checkpoint(output_dir / "checkpoints" / ("epoch_%03d.pt" % epoch), model, optimizer, epoch, global_step, cfg)
        save_checkpoint(output_dir / "checkpoints" / "latest.pt", model, optimizer, epoch, global_step, cfg)
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
