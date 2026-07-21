from __future__ import annotations

import argparse
import json

from gam_reg.data.hntsmrg24 import (
    discover_hntsmrg24_cases,
    inspect_hntsmrg24_geometry,
    prepare_hntsmrg24_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare physical-space-safe HNTS-MRG 2024 longitudinal registration pairs"
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--manifest-dir")
    parser.add_argument("--target-spacing", nargs=3, type=float, default=[1.5, 1.5, 1.5])
    parser.add_argument("--target-shape", nargs=3, type=int, default=[128, 160, 160])
    parser.add_argument("--lower-percentile", type=float, default=0.5)
    parser.add_argument("--upper-percentile", type=float, default=99.5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--prealignment",
        choices=["rigid-affine", "official-deformable"],
        default="rigid-affine",
    )
    parser.add_argument("--rigid-iterations", type=int, default=200)
    parser.add_argument("--affine-iterations", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()

    if args.inspect_only:
        cases = discover_hntsmrg24_cases(args.source_root)
        print(json.dumps(inspect_hntsmrg24_geometry(cases), indent=2, sort_keys=True))
        return
    if not args.output_root or not args.manifest_dir:
        parser.error("--output-root and --manifest-dir are required unless --inspect-only is used")
    summary = prepare_hntsmrg24_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        manifest_dir=args.manifest_dir,
        target_spacing_dhw=args.target_spacing,
        target_shape_dhw=args.target_shape,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        prealignment=args.prealignment,
        rigid_iterations=args.rigid_iterations,
        affine_iterations=args.affine_iterations,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
