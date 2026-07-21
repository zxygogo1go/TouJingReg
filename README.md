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

`moving_seg` and `fixed_seg` are optional. Images are expected to be pre-aligned and resampled to identical spacing, orientation, and shape before model input.

Supported volume formats: `.npy`, `.npz`, `.pt`, `.pth`, `.nii`, `.nii.gz`, `.nrrd`, `.seg.nrrd`.

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

## Validation And Inference

```bash
python validate.py --checkpoint runs/gam_reg/checkpoints/latest.pt --manifest pairs.csv --data-root /data/han
python infer.py --checkpoint runs/gam_reg/checkpoints/latest.pt --moving moving.npy --fixed fixed.npy --output-dir outputs/case01 --save-npy
```

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
