from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml


MANIFEST_FIELDS = ("moving", "fixed", "moving_seg", "fixed_seg")


def read_case_ids(path: str | Path) -> List[str]:
    path = Path(path)
    case_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    case_ids = [case_id for case_id in case_ids if case_id and not case_id.startswith("#")]
    if not case_ids:
        raise ValueError("case list is empty: %s" % path)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case list contains duplicate IDs: %s" % path)
    return case_ids


def read_id_pairs(path: str | Path) -> List[Tuple[str, str]]:
    path = Path(path)
    pairs: List[Tuple[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row_index, row in enumerate(csv.reader(handle), start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != 2:
                raise ValueError("pair row %d in %s must contain exactly two columns" % (row_index, path))
            moving, fixed = (value.strip() for value in row)
            if row_index == 1 and moving.lower() in {"moving", "moving_id"} and fixed.lower() in {"fixed", "fixed_id"}:
                continue
            if not moving or not fixed:
                raise ValueError("pair row %d in %s contains an empty case ID" % (row_index, path))
            pairs.append((moving, fixed))
    if not pairs:
        raise ValueError("pair file is empty: %s" % path)
    return pairs


def build_training_pairs(case_ids: Sequence[str], include_self: bool = False) -> List[Tuple[str, str]]:
    pairs = [
        (moving, fixed)
        for moving in case_ids
        for fixed in case_ids
        if include_self or moving != fixed
    ]
    if not pairs:
        raise ValueError("training split does not produce any registration pairs")
    return pairs


def _manifest_row(moving: str, fixed: str, segmentation_dir: Optional[str]) -> Dict[str, str]:
    row = {
        "moving": "images/%s.npy" % moving,
        "fixed": "images/%s.npy" % fixed,
        "moving_seg": "",
        "fixed_seg": "",
    }
    if segmentation_dir is not None:
        row["moving_seg"] = "%s/%s.npy" % (segmentation_dir, moving)
        row["fixed_seg"] = "%s/%s.npy" % (segmentation_dir, fixed)
    return row


def write_manifest(
    path: str | Path,
    pairs: Iterable[Tuple[str, str]],
    segmentation_dir: Optional[str],
) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_manifest_row(moving, fixed, segmentation_dir) for moving, fixed in pairs]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError("required dataset file is missing: %s" % path)


def _validate_metadata(
    path: Path,
    case_id: str,
    image_shape: Optional[Tuple[int, ...]],
    canonical_label_map: Dict[str, int],
    physical_fields: Dict[str, int],
    physical_values: Dict[str, set],
) -> Optional[Tuple[float, ...]]:
    _require_file(path)
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    metadata_case_id = metadata.get("case_id")
    if metadata_case_id is not None and str(metadata_case_id) != case_id:
        raise ValueError("metadata case_id mismatch in %s" % path)
    target_shape = metadata.get("target_shape")
    if image_shape is not None and target_shape is not None:
        if tuple(int(value) for value in target_shape) != image_shape:
            raise ValueError("metadata target_shape does not match image in %s" % path)
    label_map = metadata.get("label_map", {})
    if not isinstance(label_map, dict):
        raise ValueError("metadata label_map must be an object in %s" % path)
    for name, label in label_map.items():
        label = int(label)
        if name in canonical_label_map and canonical_label_map[name] != label:
            raise ValueError("inconsistent label_map entry %s in %s" % (name, path))
        canonical_label_map[name] = label
    field_groups = {
        "axis_order": ("array_axis_order",),
        "orientation": ("target_direction", "array_direction"),
        "origin_or_crop": ("target_origin", "crop_frame_origin", "crop_bbox", "crop_bounds"),
        "prealignment": ("rigid_transform", "affine_transform", "prealigned"),
    }
    for group, fields in field_groups.items():
        for field in fields:
            if field in metadata:
                physical_fields[group] += 1
                physical_values[group].add(json.dumps(metadata[field], sort_keys=True))
                break
    axis_order = metadata.get("array_axis_order")
    if axis_order is not None:
        normalized_axis_order = str(axis_order).lower().replace(",", "").replace(" ", "")
        if normalized_axis_order not in {"zyx", "dhw"}:
            raise ValueError("array_axis_order must be zyx/DHW in %s" % path)
    spacing = metadata.get("target_spacing")
    return None if spacing is None else tuple(float(value) for value in spacing)


def validate_layout(
    data_root: str | Path,
    case_ids: Sequence[str],
    segmentation_dir: Optional[str] = "seg_o",
    expected_shape: Optional[Sequence[int]] = None,
    check_arrays: bool = True,
    require_metadata: bool = True,
) -> Dict[str, object]:
    root = Path(data_root)
    unique_ids = list(dict.fromkeys(str(case_id) for case_id in case_ids))
    if not unique_ids:
        raise ValueError("no case IDs were provided for layout validation")
    expected = None if expected_shape is None else tuple(int(value) for value in expected_shape)
    common_shape = expected
    image_min = float("inf")
    image_max = float("-inf")
    labels = set()
    canonical_label_map: Dict[str, int] = {}
    target_spacings = set()
    physical_fields = {"axis_order": 0, "orientation": 0, "origin_or_crop": 0, "prealignment": 0}
    physical_values = {key: set() for key in physical_fields}

    for case_id in unique_ids:
        image_path = root / "images" / (case_id + ".npy")
        _require_file(image_path)
        seg_path = None if segmentation_dir is None else root / segmentation_dir / (case_id + ".npy")
        if seg_path is not None:
            _require_file(seg_path)

        image_shape: Optional[Tuple[int, ...]] = None
        if check_arrays:
            image = np.load(image_path, mmap_mode="r")
            image_shape = tuple(int(value) for value in image.shape)
            if image.ndim != 3:
                raise ValueError("image must be 3D in %s" % image_path)
            if not np.issubdtype(image.dtype, np.floating):
                raise ValueError("image dtype must be floating point in %s" % image_path)
            if not np.isfinite(image).all():
                raise ValueError("image contains non-finite values in %s" % image_path)
            case_min = float(image.min())
            case_max = float(image.max())
            if case_min < -1.0e-4 or case_max > 1.0001:
                raise ValueError("preprocessed image values must lie in [0, 1] in %s" % image_path)
            image_min = min(image_min, case_min)
            image_max = max(image_max, case_max)
            if common_shape is None:
                common_shape = image_shape
            if image_shape != common_shape:
                raise ValueError("inconsistent image shape in %s: expected %s, got %s" % (image_path, common_shape, image_shape))

            if seg_path is not None:
                segmentation = np.load(seg_path, mmap_mode="r")
                if tuple(segmentation.shape) != image_shape:
                    raise ValueError("segmentation shape does not match image in %s" % seg_path)
                if not np.issubdtype(segmentation.dtype, np.integer):
                    raise ValueError("segmentation dtype must be integer in %s" % seg_path)
                case_labels = np.unique(segmentation)
                if case_labels.size and int(case_labels.min()) < 0:
                    raise ValueError("segmentation labels must be non-negative in %s" % seg_path)
                labels.update(int(value) for value in case_labels.tolist())

        metadata_path = root / "metadata" / (case_id + ".json")
        if require_metadata:
            spacing = _validate_metadata(
                metadata_path,
                case_id,
                image_shape,
                canonical_label_map,
                physical_fields,
                physical_values,
            )
            if spacing is not None:
                target_spacings.add(spacing)

    warnings = []
    total = len(unique_ids)
    if require_metadata:
        if len(target_spacings) > 1:
            raise ValueError("metadata contains inconsistent target_spacing values")
        if not target_spacings:
            warnings.append("metadata does not record target_spacing")
        if physical_fields["axis_order"] < total:
            warnings.append("metadata cannot prove that every array uses z,y,x (D,H,W) axis order")
        elif len(physical_values["axis_order"]) > 1:
            raise ValueError("metadata contains inconsistent array_axis_order values")
        if physical_fields["orientation"] < total:
            warnings.append("metadata cannot prove a common orientation/direction for every case")
        elif len(physical_values["orientation"]) > 1:
            raise ValueError("metadata contains inconsistent target orientation/direction values")
        if physical_fields["origin_or_crop"] < total:
            warnings.append("metadata cannot prove a common physical origin or crop frame for every case")
        elif len(physical_values["origin_or_crop"]) > 1:
            warnings.append("metadata records different target origins or crop frames across cases")
        if physical_fields["prealignment"] < total:
            warnings.append("metadata cannot prove rigid/affine pre-alignment for every case")
    if labels and labels != set(range(max(labels) + 1)):
        warnings.append("segmentation labels are not contiguous across the referenced cases")
    if segmentation_dir is not None and labels == {0}:
        warnings.append("segmentation contains background only across all referenced cases")

    label_values = set(labels)
    label_values.update(int(value) for value in canonical_label_map.values())
    return {
        "num_cases": total,
        "shape_dhw": None if common_shape is None else list(common_shape),
        "target_spacings": [list(values) for values in sorted(target_spacings)],
        "image_value_range": None if not check_arrays else [image_min, image_max],
        "segmentation_dir": segmentation_dir,
        "labels": sorted(labels),
        "num_anatomy_classes": None if not label_values else max(label_values) + 1,
        "label_map": dict(sorted(canonical_label_map.items())),
        "warnings": warnings,
    }


def prepare_layout_manifests(
    data_root: str | Path,
    output_dir: str | Path,
    split_dir: Optional[str | Path] = None,
    segmentation_dir: Optional[str] = "seg_o",
    include_self: bool = False,
    expected_shape: Optional[Sequence[int]] = None,
    check_arrays: bool = True,
    require_metadata: bool = True,
) -> Dict[str, object]:
    root = Path(data_root)
    split_root = root / "lists" / "paper_split" if split_dir is None else Path(split_dir)
    train_ids = read_case_ids(split_root / "trn_list_inter.txt")
    val_pairs = read_id_pairs(split_root / "val_pairs.csv")
    test_pairs = read_id_pairs(split_root / "test_pairs.csv")
    train_cases = set(train_ids)
    val_cases = {case_id for pair in val_pairs for case_id in pair}
    test_cases = {case_id for pair in test_pairs for case_id in pair}
    split_overlaps = {
        "train/val": train_cases & val_cases,
        "train/test": train_cases & test_cases,
        "val/test": val_cases & test_cases,
    }
    for split_names, overlap in split_overlaps.items():
        if overlap:
            raise ValueError("case-level split leakage in %s: %s" % (split_names, sorted(overlap)))
    train_pairs = build_training_pairs(train_ids, include_self=include_self)
    referenced_ids = train_ids + [case_id for pair in val_pairs + test_pairs for case_id in pair]

    summary = validate_layout(
        root,
        referenced_ids,
        segmentation_dir=segmentation_dir,
        expected_shape=expected_shape,
        check_arrays=check_arrays,
        require_metadata=require_metadata,
    )
    output_root = Path(output_dir)
    counts = {
        "train_pairs": write_manifest(output_root / "train_pairs.csv", train_pairs, segmentation_dir),
        "val_pairs": write_manifest(output_root / "val_pairs.csv", val_pairs, segmentation_dir),
        "test_pairs": write_manifest(output_root / "test_pairs.csv", test_pairs, segmentation_dir),
    }
    summary.update(
        {
            "data_root": str(root.resolve()),
            "split_dir": str(split_root.resolve()),
            "include_self_pairs": bool(include_self),
            "manifest_counts": counts,
        }
    )
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    num_classes = summary["num_anatomy_classes"]
    use_segmentation = segmentation_dir is not None and num_classes is not None
    dataset_config = {
        "model": {
            "use_anatomy_head": use_segmentation,
            "num_anatomy_classes": int(num_classes) if use_segmentation else 0,
        },
        "loss": {
            "weights": {
                "dice": 1.0 if use_segmentation else 0.0,
                "anatomy": 0.1 if use_segmentation else 0.0,
            }
        },
        "data": {
            "image_normalization": "zero_one",
            "target_shape": summary["shape_dhw"],
        },
    }
    with (output_root / "dataset_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dataset_config, handle, sort_keys=False)
    return summary
