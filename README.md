# GAM-Reg

GAM-Reg (Gaussian Anatomy Matching Registration) is a PyTorch implementation for 3D deformable head-and-neck registration. The default direction is moving-to-fixed:

- `phi_fwd = phi_m2f = exp(v)` maps moving coordinates to fixed coordinates.
- `phi_inv = phi_f2m = exp(-v)` maps fixed coordinates back to moving coordinates.
- `warped_moving = spatial_transform(moving, phi_inv)` is the image resampled on the fixed grid.
- All transform grids use normalized xyz coordinates in `[-1, 1]`; `grid[..., 0]` is x, `grid[..., 1]` is y, and `grid[..., 2]` is z.

## Layout

```text
gam_reg/
  configs/gam_reg_stable.yaml
  models/
  losses/
  data/
  metrics/
tests/
train.py
validate.py
infer.py
prepare_dataset.py
evaluate_ablation.py
```

The implementation includes Gaussian W2, log-domain Sinkhorn, chunked Gaussian-to-volume propagation, and scaling-and-squaring integration.

## Linux Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA servers, install the PyTorch wheel matching the server CUDA version from the official PyTorch instructions, then install the remaining requirements.

## Tests

The tests are `unittest` compatible and also discoverable by pytest.

```bash
python -m unittest discover -s tests
```

On this macOS workspace, the available PyTorch is x86_64, so local tests were run with:

```bash
arch -x86_64 /usr/bin/python3 -m unittest discover -s tests
```

## Data Manifest

Real-pair training uses a CSV:

```csv
moving,fixed,moving_seg,fixed_seg
case01_moving.npy,case01_fixed.npy,case01_moving_seg.npy,case01_fixed_seg.npy
```

`moving_seg` and `fixed_seg` are optional. Images are expected to be pre-aligned and resampled to identical spacing, orientation, axis order, and shape before model input.

Supported volume formats: `.npy`, `.npz`, `.pt`, `.pth`, `.nii`, `.nii.gz`, `.nrrd`, `.seg.nrrd`.

Raw CT volumes use the default `hu` normalization. Preprocessed arrays already in `[0,1]` must use `--image-normalization zero_one` or set `data.image_normalization: zero_one` in the YAML config.

### Reusing the MUSA server layout

For a dataset organized as `images/`, `seg_o/`, `seg_b/`, `metadata/`, and `lists/paper_split/`, validate it and convert its ID-based splits before training:

```bash
python prepare_dataset.py \
  --data-root /root/autodl-tmp/MUSA/data_hanseg \
  --output-dir manifests/hanseg \
  --seg-dir seg_o \
  --expected-shape 160 160 192
```

This creates train/validation/test manifests, `dataset_summary.json`, and a `dataset_config.yaml` containing the detected anatomy class count and data normalization settings. Review every warning in the summary, especially missing axis-order, orientation, crop-frame, or rigid/affine pre-alignment evidence. See `linux_server_dataset_layout.md` for the complete server workflow.

## Training

Synthetic warm-up smoke training:

```bash
python train.py --stage synthetic-warmup --epochs 1 --steps-per-epoch 100 --output-dir runs/synthetic
```

Fast local sanity check:

```bash
python train.py --config gam_reg/configs/gam_reg_tiny_demo.yaml --stage synthetic-warmup --epochs 1 --steps-per-epoch 1 --synthetic-shape 16 16 16 --output-dir runs/tiny_demo
```

Real registration warm-up or joint training:

```bash
python train.py --stage joint --manifest pairs.csv --data-root /data/han --output-dir runs/gam_reg
```

Resume:

```bash
python train.py --stage joint --manifest pairs.csv --resume runs/gam_reg/checkpoints/latest.pt
```

`--resume` continues the same training stage, including its optimizer and stage-local schedule. To transition between stages, load model weights only:

```bash
python train.py --stage registration-warmup --manifest pairs.csv --init-checkpoint runs/synthetic/checkpoints/latest.pt
```

During `registration-warmup`, anchor and Jacobian weights ramp from their configured starting values to the target loss weights over `training.stage_schedules.registration-warmup.ramp_steps` successful optimizer steps. Smoothness is computed from physical displacement gradients using `data.spacing_dhw`.

## Validation And Inference

```bash
python validate.py --checkpoint runs/gam_reg/checkpoints/latest.pt --manifest pairs.csv --data-root /data/han
python infer.py --checkpoint runs/gam_reg/checkpoints/latest.pt --moving moving.npy --fixed fixed.npy --output-dir outputs/case01 --save-npy
```

Validation reports both aggregate and per-pair metrics, including folding ratio, the fraction below the configured safe Jacobian determinant, and the minimum determinant. Use `--output-json path/to/report.json` to save the complete report.

## Ablations

Available variants:

- `baseline_unet_registration`
- `point_tokens`
- `isotropic_gaussian`
- `anisotropic_gaussian_without_w2`
- `anisotropic_gaussian_w2_no_sinkhorn`
- `full`
- `full_without_anchor_loss`
- `full_without_dice`

Run:

```bash
python evaluate_ablation.py --manifest pairs.csv --data-root /data/han --checkpoint-dir checkpoints/ablations
```

The checkpoint directory may contain files named `<variant>.pt`.
