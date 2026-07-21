from __future__ import annotations

import argparse
import json

from gam_reg.data.dataset_layout import prepare_layout_manifests


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a MUSA-style preprocessed dataset and create GAM-Reg manifests"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-dir")
    parser.add_argument("--seg-dir", choices=["seg_o", "seg_b", "none"], default="seg_o")
    parser.add_argument("--include-self", action="store_true")
    parser.add_argument("--expected-shape", nargs=3, type=int)
    parser.add_argument("--skip-array-check", action="store_true")
    parser.add_argument("--allow-missing-metadata", action="store_true")
    args = parser.parse_args()

    summary = prepare_layout_manifests(
        data_root=args.data_root,
        output_dir=args.output_dir,
        split_dir=args.split_dir,
        segmentation_dir=None if args.seg_dir == "none" else args.seg_dir,
        include_self=args.include_self,
        expected_shape=args.expected_shape,
        check_arrays=not args.skip_array_check,
        require_metadata=not args.allow_missing_metadata,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
