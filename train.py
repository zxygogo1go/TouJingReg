from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from gam_reg.amp import make_grad_scaler, nonfinite_gradient_names, require_finite
from gam_reg.config import load_config
from gam_reg.data.dataset import SyntheticRegistrationDataset, VolumePairDataset
from gam_reg.losses.total_loss import TotalRegistrationLoss
from gam_reg.models.gam_reg import GAMReg
from gam_reg.training_schedule import stage_loss_weights


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
        spacing_dhw=cfg.get("data", {}).get("spacing_dhw", [1.0, 1.0, 1.0]),
    )


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    step: int,
    stage: str,
    stage_step: int,
    cfg: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "stage": stage,
            "stage_step": stage_step,
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
    parser.add_argument("--init-checkpoint")
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
    if args.resume and args.init_checkpoint:
        parser.error("--resume and --init-checkpoint are mutually exclusive")

    torch.manual_seed(args.seed)
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    if args.image_normalization is None:
        args.image_normalization = data_cfg.get("image_normalization", "hu")
    if args.target_shape is None:
        args.target_shape = data_cfg.get("target_shape")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = GAMReg(cfg).to(device)
    loss_fn = TotalRegistrationLoss(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    start_epoch = 0
    global_step = 0
    stage_step = 0
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        checkpoint_stage = ckpt.get("stage", args.stage)
        if checkpoint_stage != args.stage:
            raise ValueError(
                "checkpoint stage %s does not match requested stage %s; use --init-checkpoint for stage transitions"
                % (checkpoint_stage, args.stage)
            )
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("step", 0))
        stage_step = int(ckpt.get("stage_step", global_step))

    dataset = build_dataset(args, cfg)
    loader = DataLoader(dataset, batch_size=int(cfg["training"]["batch_size"]), shuffle=True, num_workers=0)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(args.output_dir)
    except Exception:
        writer = None

    use_amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    scaler = make_grad_scaler(
        enabled=use_amp,
        init_scale=float(cfg["training"].get("amp_init_scale", 1024.0)),
        growth_interval=int(cfg["training"].get("amp_growth_interval", 2000)),
    )
    amp_max_retries = int(cfg["training"].get("amp_max_retries", 8))
    model.train()
    output_dir = Path(args.output_dir)
    for epoch in range(start_epoch, int(args.epochs)):
        running = {}
        running_weights = {}
        loader_iter = iter(loader)
        for step in range(int(args.steps_per_epoch)):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            batch = to_device(batch, device)
            effective_weights = stage_loss_weights(
                loss_fn.weights,
                stage=args.stage,
                stage_step=stage_step,
                training_config=cfg["training"],
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            amp_retries = 0
            while True:
                optimizer.zero_grad(set_to_none=True)
                first_update = global_step == 0
                velocity_head_before = None
                if first_update:
                    velocity_head_before = model.decoder.velocity_head.weight.detach().clone()
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(
                        batch["moving"],
                        batch["fixed"],
                        batch.get("moving_seg"),
                        batch.get("fixed_seg"),
                        return_debug=True,
                    )
                    total, components = loss_fn(
                        outputs,
                        fixed=batch["fixed"],
                        moving=batch["moving"],
                        moving_seg=batch.get("moving_seg"),
                        fixed_seg=batch.get("fixed_seg"),
                        spacing_dhw=batch.get("spacing_dhw"),
                        weights_override=effective_weights,
                    )
                require_finite("total loss", total)
                for component_name, component_value in components.items():
                    require_finite("loss component %s" % component_name, component_value)
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                bad_gradients = nonfinite_gradient_names(model.named_parameters())
                if bad_gradients:
                    if not use_amp:
                        raise FloatingPointError(
                            "non-finite gradients without AMP: %s" % ", ".join(bad_gradients[:20])
                        )
                    scale_before = float(scaler.get_scale())
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = float(scaler.get_scale())
                    amp_retries += 1
                    print(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "step": step,
                                "event": "amp_backoff",
                                "retry": amp_retries,
                                "scale_before": scale_before,
                                "scale_after": scale_after,
                                "nonfinite_gradients": bad_gradients[:20],
                            }
                        )
                    )
                    if scale_after >= scale_before:
                        raise FloatingPointError("GradScaler did not reduce its scale after gradient overflow")
                    if amp_retries > amp_max_retries:
                        raise FloatingPointError(
                            "AMP gradients remained non-finite after %d retries; affected parameters: %s"
                            % (amp_max_retries, ", ".join(bad_gradients[:20]))
                        )
                    del outputs, total, components, component_value, bad_gradients
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(cfg["training"]["gradient_clip_norm"]),
                )
                require_finite("gradient norm", grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scale_after = float(scaler.get_scale())
                if velocity_head_before is not None and torch.equal(
                    velocity_head_before,
                    model.decoder.velocity_head.weight.detach(),
                ):
                    raise RuntimeError("optimizer step did not update decoder.velocity_head.weight")
                break
            global_step += 1
            stage_step += 1
            for key, value in components.items():
                scalar = float(value.detach().cpu())
                running[key] = running.get(key, 0.0) + scalar
                if writer is not None:
                    writer.add_scalar("train/" + key, scalar, global_step)
            for key, value in effective_weights.items():
                running_weights[key] = running_weights.get(key, 0.0) + float(value)
                if writer is not None:
                    writer.add_scalar("weights/" + key, float(value), global_step)
            if step % 10 == 0:
                peak_allocated_gib = None
                peak_reserved_gib = None
                if device.type == "cuda":
                    peak_allocated_gib = torch.cuda.max_memory_allocated(device) / float(1024 ** 3)
                    peak_reserved_gib = torch.cuda.max_memory_reserved(device) / float(1024 ** 3)
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step,
                            "loss": float(total.detach().cpu()),
                            "grad_norm": float(grad_norm.detach().cpu()),
                            "amp_scale": scale_after,
                            "amp_retries": amp_retries,
                            "optimizer_step_skipped": False,
                            "stage": args.stage,
                            "stage_step": stage_step,
                            "effective_weights": effective_weights,
                            "cuda_peak_allocated_gib": peak_allocated_gib,
                            "cuda_peak_reserved_gib": peak_reserved_gib,
                        }
                    )
                )
            optimizer.zero_grad(set_to_none=True)
            del outputs, total, components, batch, grad_norm, component_value, value
        means = {k: v / float(args.steps_per_epoch) for k, v in running.items()}
        mean_weights = {k: v / float(args.steps_per_epoch) for k, v in running_weights.items()}
        print(json.dumps({"epoch": epoch, "mean": means, "mean_weights": mean_weights}, indent=2))
        save_checkpoint(
            output_dir / "checkpoints" / ("epoch_%03d.pt" % epoch),
            model,
            optimizer,
            epoch,
            global_step,
            args.stage,
            stage_step,
            cfg,
        )
        save_checkpoint(
            output_dir / "checkpoints" / "latest.pt",
            model,
            optimizer,
            epoch,
            global_step,
            args.stage,
            stage_step,
            cfg,
        )
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
