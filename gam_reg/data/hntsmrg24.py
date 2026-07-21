from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import nibabel as nib
import numpy as np
import yaml
from nibabel.processing import resample_from_to


@dataclass(frozen=True)
class HNTSMRG24Case:
    patient_id: str
    pre_image: Path
    pre_mask: Path
    mid_image: Path
    mid_mask: Path
    registered_pre_image: Path
    registered_pre_mask: Path

    @property
    def registration_inputs(self) -> Dict[str, Path]:
        return {
            "moving": self.registered_pre_image,
            "moving_seg": self.registered_pre_mask,
            "fixed": self.mid_image,
            "fixed_seg": self.mid_mask,
        }


def _numeric_patient_key(path: Path) -> int:
    try:
        return int(path.name)
    except ValueError as exc:
        raise ValueError("HNTS-MRG patient directory must use a numeric ID: %s" % path) from exc


def discover_hntsmrg24_cases(source_root: str | Path) -> List[HNTSMRG24Case]:
    root = Path(source_root)
    if not root.is_dir():
        raise FileNotFoundError("HNTS-MRG source root does not exist: %s" % root)
    patient_dirs = sorted((path for path in root.iterdir() if path.is_dir()), key=_numeric_patient_key)
    if not patient_dirs:
        raise ValueError("HNTS-MRG source root contains no patient directories: %s" % root)

    cases: List[HNTSMRG24Case] = []
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        case = HNTSMRG24Case(
            patient_id=patient_id,
            pre_image=patient_dir / "preRT" / (patient_id + "_preRT_T2.nii.gz"),
            pre_mask=patient_dir / "preRT" / (patient_id + "_preRT_mask.nii.gz"),
            mid_image=patient_dir / "midRT" / (patient_id + "_midRT_T2.nii.gz"),
            mid_mask=patient_dir / "midRT" / (patient_id + "_midRT_mask.nii.gz"),
            registered_pre_image=patient_dir
            / "midRT"
            / (patient_id + "_preRT_T2_registered.nii.gz"),
            registered_pre_mask=patient_dir
            / "midRT"
            / (patient_id + "_preRT_mask_registered.nii.gz"),
        )
        for path in (
            case.pre_image,
            case.pre_mask,
            case.mid_image,
            case.mid_mask,
            case.registered_pre_image,
            case.registered_pre_mask,
        ):
            if not path.is_file():
                raise FileNotFoundError("required HNTS-MRG file is missing: %s" % path)
        cases.append(case)
    return cases


def _header_record(path: Path) -> Dict[str, object]:
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError("HNTS-MRG volume must be 3D: %s" % path)
    affine = np.asarray(image.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("HNTS-MRG volume has an invalid affine: %s" % path)
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    if not np.isfinite(spacing).all() or np.any(spacing <= 0.0):
        raise ValueError("HNTS-MRG volume has invalid spacing: %s" % path)
    return {
        "shape_xyz": tuple(int(value) for value in image.shape),
        "spacing_xyz": tuple(float(value) for value in spacing),
        "orientation": tuple(str(value) for value in nib.aff2axcodes(affine)),
        "affine": affine,
    }


def inspect_hntsmrg24_geometry(
    cases: Sequence[HNTSMRG24Case],
    affine_tolerance: float = 1.0e-4,
) -> Dict[str, object]:
    if not cases:
        raise ValueError("at least one HNTS-MRG case is required")
    records = []
    orientations = set()
    for case in cases:
        pre_headers = {
            "pre": _header_record(case.pre_image),
            "pre_seg": _header_record(case.pre_mask),
        }
        if pre_headers["pre_seg"]["shape_xyz"] != pre_headers["pre"]["shape_xyz"]:
            raise ValueError("pre-RT image/mask shape mismatch for patient %s" % case.patient_id)
        if not np.allclose(
            pre_headers["pre_seg"]["affine"],
            pre_headers["pre"]["affine"],
            atol=float(affine_tolerance),
            rtol=0.0,
        ):
            raise ValueError("pre-RT image/mask affine mismatch for patient %s" % case.patient_id)

        headers = {name: _header_record(path) for name, path in case.registration_inputs.items()}
        fixed = headers["fixed"]
        for name, header in headers.items():
            if header["shape_xyz"] != fixed["shape_xyz"]:
                raise ValueError(
                    "registered geometry shape mismatch for patient %s (%s)" % (case.patient_id, name)
                )
            if not np.allclose(
                header["affine"],
                fixed["affine"],
                atol=float(affine_tolerance),
                rtol=0.0,
            ):
                raise ValueError(
                    "registered geometry affine mismatch for patient %s (%s)" % (case.patient_id, name)
                )
        orientations.add(fixed["orientation"])
        records.append(
            {
                "patient_id": case.patient_id,
                "shape_xyz": fixed["shape_xyz"],
                "spacing_xyz": fixed["spacing_xyz"],
                "orientation": fixed["orientation"],
            }
        )
    if len(orientations) != 1:
        raise ValueError("HNTS-MRG fixed images do not share one orientation: %s" % sorted(orientations))

    shapes = np.asarray([record["shape_xyz"] for record in records], dtype=np.int64)
    spacings = np.asarray([record["spacing_xyz"] for record in records], dtype=np.float64)
    extents = (shapes - 1) * spacings
    return {
        "num_patients": len(records),
        "orientation": list(next(iter(orientations))),
        "shape_xyz": {
            "minimum": shapes.min(axis=0).tolist(),
            "median": np.median(shapes, axis=0).tolist(),
            "maximum": shapes.max(axis=0).tolist(),
        },
        "spacing_xyz": {
            "minimum": spacings.min(axis=0).tolist(),
            "median": np.median(spacings, axis=0).tolist(),
            "maximum": spacings.max(axis=0).tolist(),
        },
        "extent_mm_xyz": {
            "minimum": extents.min(axis=0).tolist(),
            "median": np.median(extents, axis=0).tolist(),
            "maximum": extents.max(axis=0).tolist(),
        },
    }


def _centered_target_geometry(
    fixed_image: nib.spatialimages.SpatialImage,
    target_spacing_dhw: Sequence[float],
    target_shape_dhw: Sequence[int],
) -> Tuple[Tuple[int, int, int], np.ndarray]:
    spacing_dhw = np.asarray(target_spacing_dhw, dtype=np.float64)
    shape_dhw = np.asarray(target_shape_dhw, dtype=np.int64)
    if spacing_dhw.shape != (3,) or not np.isfinite(spacing_dhw).all() or np.any(spacing_dhw <= 0.0):
        raise ValueError("target_spacing_dhw must contain three finite positive values")
    if shape_dhw.shape != (3,) or np.any(shape_dhw < 3):
        raise ValueError("target_shape_dhw must contain three values of at least three")

    source_linear = np.asarray(fixed_image.affine[:3, :3], dtype=np.float64)
    source_spacing = np.linalg.norm(source_linear, axis=0)
    if np.any(source_spacing <= 0.0):
        raise ValueError("fixed image affine has a degenerate spatial axis")
    direction = source_linear / source_spacing
    target_spacing_xyz = spacing_dhw[::-1]
    target_shape_xyz = tuple(int(value) for value in shape_dhw[::-1])
    target_linear = direction * target_spacing_xyz

    source_center_voxel = (np.asarray(fixed_image.shape, dtype=np.float64) - 1.0) / 2.0
    source_center_world = nib.affines.apply_affine(fixed_image.affine, source_center_voxel)
    target_center_voxel = (np.asarray(target_shape_xyz, dtype=np.float64) - 1.0) / 2.0
    target_affine = np.eye(4, dtype=np.float64)
    target_affine[:3, :3] = target_linear
    target_affine[:3, 3] = source_center_world - target_linear @ target_center_voxel
    return target_shape_xyz, target_affine


def _mask_bbox_fits_target(
    mask_image: nib.spatialimages.SpatialImage,
    target_shape_xyz: Sequence[int],
    target_affine: np.ndarray,
    tolerance_voxels: float = 0.51,
) -> bool:
    mask = np.asanyarray(mask_image.dataobj)
    foreground = np.argwhere(mask > 0)
    if foreground.size == 0:
        return True
    lower = foreground.min(axis=0)
    upper = foreground.max(axis=0)
    corners = np.asarray(np.meshgrid(*zip(lower, upper), indexing="ij")).reshape(3, -1).T
    world = nib.affines.apply_affine(mask_image.affine, corners)
    target_voxels = nib.affines.apply_affine(np.linalg.inv(target_affine), world)
    target_shape = np.asarray(target_shape_xyz, dtype=np.float64)
    return bool(
        np.all(target_voxels >= -float(tolerance_voxels))
        and np.all(target_voxels <= target_shape - 1.0 + float(tolerance_voxels))
    )


def robust_normalize_mri(
    volume: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> Tuple[np.ndarray, Dict[str, float]]:
    array = np.asarray(volume, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("MRI volume must be 3D")
    if not np.isfinite(array).all():
        raise ValueError("MRI volume contains non-finite values")
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("MRI normalization percentiles are invalid")
    samples = array[np.abs(array) > 1.0e-6]
    if samples.size < 100:
        samples = array.reshape(-1)
    lower, upper = np.percentile(samples, [lower_percentile, upper_percentile]).astype(np.float64)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("MRI volume has a degenerate robust intensity range")
    normalized = np.clip((array - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)
    return normalized, {
        "lower_percentile": float(lower_percentile),
        "upper_percentile": float(upper_percentile),
        "lower_value": float(lower),
        "upper_value": float(upper),
    }


def _resample_image(
    image: nib.spatialimages.SpatialImage,
    target_shape_xyz: Sequence[int],
    target_affine: np.ndarray,
    order: int,
) -> np.ndarray:
    resampled = resample_from_to(
        image,
        (tuple(int(value) for value in target_shape_xyz), target_affine),
        order=int(order),
        mode="constant",
        cval=0.0,
    )
    return np.asarray(resampled.dataobj, dtype=np.float32)


def _registration_sampling_percentage(num_voxels: int) -> float:
    return min(1.0, max(0.02, 10000.0 / max(int(num_voxels), 1)))


def _configure_registration_method(
    iterations: int,
    seed: int,
    num_voxels: int,
    learning_rate: float,
):
    import SimpleITK as sitk

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    sampling_percentage = _registration_sampling_percentage(num_voxels)
    if sampling_percentage >= 1.0:
        registration.SetMetricSamplingStrategy(registration.NONE)
    else:
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(sampling_percentage, int(seed))
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=float(learning_rate),
        minStep=1.0e-4,
        numberOfIterations=int(iterations),
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1.0e-8,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2.0, 1.0, 0.0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return registration


def rigid_affine_prealign_case(
    case: HNTSMRG24Case,
    seed: int = 1234,
    rigid_iterations: int = 200,
    affine_iterations: int = 200,
) -> Tuple[nib.Nifti1Image, nib.Nifti1Image, Dict[str, object]]:
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError(
            "SimpleITK is required for HNTS-MRG rigid-affine prealignment; install project requirements"
        ) from exc

    if rigid_iterations <= 0 or affine_iterations <= 0:
        raise ValueError("registration iteration counts must be positive")
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    fixed = sitk.Cast(sitk.ReadImage(str(case.mid_image)), sitk.sitkFloat32)
    moving = sitk.Cast(sitk.ReadImage(str(case.pre_image)), sitk.sitkFloat32)
    moving_mask = sitk.ReadImage(str(case.pre_mask))
    if fixed.GetDimension() != 3 or moving.GetDimension() != 3 or moving_mask.GetDimension() != 3:
        raise ValueError("HNTS-MRG rigid-affine registration requires 3D volumes")

    fixed_metric = sitk.Normalize(fixed)
    moving_metric = sitk.Normalize(moving)
    rigid = sitk.CenteredTransformInitializer(
        fixed_metric,
        moving_metric,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    rigid_registration = _configure_registration_method(
        rigid_iterations,
        seed,
        fixed_metric.GetNumberOfPixels(),
        learning_rate=2.0,
    )
    rigid_registration.SetInitialTransform(rigid, inPlace=True)
    rigid_metric_before = float(rigid_registration.MetricEvaluate(fixed_metric, moving_metric))
    rigid_registration.Execute(fixed_metric, moving_metric)
    rigid_metric = float(rigid_registration.GetMetricValue())
    rigid_metric_after = float(rigid_registration.MetricEvaluate(fixed_metric, moving_metric))
    if not np.isfinite([rigid_metric_before, rigid_metric, rigid_metric_after]).all():
        raise FloatingPointError("non-finite rigid registration metric for patient %s" % case.patient_id)
    if rigid_metric_after > rigid_metric_before + 1.0e-4:
        raise ValueError("rigid registration worsened full-volume MI for patient %s" % case.patient_id)

    affine = sitk.AffineTransform(3)
    affine.SetCenter(rigid.GetCenter())
    affine.SetMatrix(rigid.GetMatrix())
    affine.SetTranslation(rigid.GetTranslation())
    affine_registration = _configure_registration_method(
        affine_iterations,
        seed + 1,
        fixed_metric.GetNumberOfPixels(),
        learning_rate=0.1,
    )
    affine_registration.SetInitialTransform(affine, inPlace=True)
    affine_metric_before = float(affine_registration.MetricEvaluate(fixed_metric, moving_metric))
    affine_registration.Execute(fixed_metric, moving_metric)
    affine_metric = float(affine_registration.GetMetricValue())
    affine_metric_after = float(affine_registration.MetricEvaluate(fixed_metric, moving_metric))
    if not np.isfinite([affine_metric_before, affine_metric, affine_metric_after]).all():
        raise FloatingPointError("non-finite affine registration metric for patient %s" % case.patient_id)

    matrix = np.asarray(affine.GetMatrix(), dtype=np.float64).reshape(3, 3)
    translation = np.asarray(affine.GetTranslation(), dtype=np.float64)
    determinant = float(np.linalg.det(matrix))
    if not np.isfinite(matrix).all() or not np.isfinite(translation).all():
        raise FloatingPointError("non-finite affine transform for patient %s" % case.patient_id)
    fallback_reasons = []
    if not 0.5 <= determinant <= 2.0:
        fallback_reasons.append("affine determinant %.6f is outside [0.5,2.0]" % determinant)
    if float(np.linalg.norm(translation)) > 250.0:
        fallback_reasons.append("affine translation exceeds 250 mm")
    if affine_metric_after > affine_metric_before + 1.0e-4:
        fallback_reasons.append("affine full-volume MI worsened")
    selected_transform = rigid if fallback_reasons else affine
    selected_stage = "rigid" if fallback_reasons else "affine"

    registered_moving = sitk.Resample(
        moving,
        fixed,
        selected_transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    registered_mask = sitk.Resample(
        moving_mask,
        fixed,
        selected_transform,
        sitk.sitkNearestNeighbor,
        0,
        moving_mask.GetPixelID(),
    )
    fixed_nib = nib.load(str(case.mid_image))
    moving_xyz = np.transpose(sitk.GetArrayFromImage(registered_moving), (2, 1, 0)).astype(
        np.float32,
        copy=False,
    )
    mask_xyz = np.transpose(sitk.GetArrayFromImage(registered_mask), (2, 1, 0))
    moving_nib = nib.Nifti1Image(moving_xyz, fixed_nib.affine)
    mask_nib = nib.Nifti1Image(mask_xyz, fixed_nib.affine)
    record: Dict[str, object] = {
        "method": "SimpleITK centered rigid followed by affine Mattes-MI registration",
        "seed": int(seed),
        "sampling_percentage": _registration_sampling_percentage(fixed_metric.GetNumberOfPixels()),
        "histogram_bins": 50,
        "shrink_factors": [4, 2, 1],
        "smoothing_sigmas_mm": [2.0, 1.0, 0.0],
        "rigid_iterations": int(rigid_iterations),
        "rigid_full_metric_before": rigid_metric_before,
        "rigid_final_metric": rigid_metric,
        "rigid_full_metric_after": rigid_metric_after,
        "rigid_stop_condition": rigid_registration.GetOptimizerStopConditionDescription(),
        "affine_iterations": int(affine_iterations),
        "affine_full_metric_before": affine_metric_before,
        "affine_final_metric": affine_metric,
        "affine_full_metric_after": affine_metric_after,
        "affine_stop_condition": affine_registration.GetOptimizerStopConditionDescription(),
        "affine_center_lps": list(affine.GetCenter()),
        "affine_matrix_lps": matrix.tolist(),
        "affine_translation_lps": translation.tolist(),
        "affine_determinant": determinant,
        "selected_stage": selected_stage,
        "fallback_reasons": fallback_reasons,
    }
    return moving_nib, mask_nib, record


def _validate_and_convert_mask(mask: np.ndarray, source_path: Path) -> np.ndarray:
    rounded = np.rint(mask)
    if not np.allclose(mask, rounded, atol=1.0e-5, rtol=0.0):
        raise ValueError("resampled mask contains non-integer values: %s" % source_path)
    labels = set(int(value) for value in np.unique(rounded).tolist())
    if not labels.issubset({0, 1, 2}):
        raise ValueError("HNTS-MRG mask labels must be a subset of {0,1,2}: %s" % source_path)
    return rounded.astype(np.int16)


def _save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)


def preprocess_hntsmrg24_case(
    case: HNTSMRG24Case,
    output_root: str | Path,
    target_spacing_dhw: Sequence[float] = (1.5, 1.5, 1.5),
    target_shape_dhw: Sequence[int] = (128, 160, 160),
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    prealignment: str = "rigid-affine",
    registration_seed: int = 1234,
    rigid_iterations: int = 200,
    affine_iterations: int = 200,
    overwrite: bool = False,
) -> Dict[str, object]:
    output_root = Path(output_root)
    output_id = "hntsmrg24_%03d" % int(case.patient_id)
    output_paths = {
        "moving": output_root / "images" / (output_id + "_pre_aligned.npy"),
        "fixed": output_root / "images" / (output_id + "_mid.npy"),
        "moving_seg": output_root / "seg_o" / (output_id + "_pre_aligned.npy"),
        "fixed_seg": output_root / "seg_o" / (output_id + "_mid.npy"),
        "metadata": output_root / "metadata" / (output_id + ".json"),
    }
    if not overwrite:
        existing = [str(path) for path in output_paths.values() if path.exists()]
        if existing:
            raise FileExistsError("refusing to overwrite prepared HNTS-MRG files: %s" % existing)

    prealignment = str(prealignment).lower()
    if prealignment == "rigid-affine":
        moving_image, moving_mask, prealignment_record = rigid_affine_prealign_case(
            case,
            seed=int(registration_seed),
            rigid_iterations=int(rigid_iterations),
            affine_iterations=int(affine_iterations),
        )
        source_images = {
            "moving": moving_image,
            "moving_seg": moving_mask,
            "fixed": nib.load(str(case.mid_image)),
            "fixed_seg": nib.load(str(case.mid_mask)),
        }
        moving_source_path = case.pre_mask
    elif prealignment == "official-deformable":
        source_images = {name: nib.load(str(path)) for name, path in case.registration_inputs.items()}
        prealignment_record = {
            "method": "official HNTS-MRG registered pre-RT volume (may include deformable Elastix)",
        }
        moving_source_path = case.registered_pre_mask
    else:
        raise ValueError("prealignment must be 'rigid-affine' or 'official-deformable'")
    fixed_image = source_images["fixed"]
    for name, image in source_images.items():
        if image.shape != fixed_image.shape or not np.allclose(
            image.affine, fixed_image.affine, atol=1.0e-4, rtol=0.0
        ):
            raise ValueError(
                "registered geometry mismatch for patient %s (%s)" % (case.patient_id, name)
            )

    target_shape_xyz, target_affine = _centered_target_geometry(
        fixed_image,
        target_spacing_dhw=target_spacing_dhw,
        target_shape_dhw=target_shape_dhw,
    )
    for name in ("moving_seg", "fixed_seg"):
        if not _mask_bbox_fits_target(source_images[name], target_shape_xyz, target_affine):
            raise ValueError(
                "target ROI would crop tumor labels for patient %s (%s)" % (case.patient_id, name)
            )

    moving_xyz = _resample_image(source_images["moving"], target_shape_xyz, target_affine, order=1)
    fixed_xyz = _resample_image(source_images["fixed"], target_shape_xyz, target_affine, order=1)
    moving_seg_xyz = _validate_and_convert_mask(
        _resample_image(source_images["moving_seg"], target_shape_xyz, target_affine, order=0),
        moving_source_path,
    )
    fixed_seg_xyz = _validate_and_convert_mask(
        _resample_image(source_images["fixed_seg"], target_shape_xyz, target_affine, order=0),
        case.mid_mask,
    )
    moving_xyz, moving_normalization = robust_normalize_mri(
        moving_xyz,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    fixed_xyz, fixed_normalization = robust_normalize_mri(
        fixed_xyz,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )

    source_labels = {
        name: sorted(int(value) for value in np.unique(np.asanyarray(source_images[name].dataobj)).tolist())
        for name in ("moving_seg", "fixed_seg")
    }
    output_labels = {
        "moving_seg": sorted(int(value) for value in np.unique(moving_seg_xyz).tolist()),
        "fixed_seg": sorted(int(value) for value in np.unique(fixed_seg_xyz).tolist()),
    }
    for name in ("moving_seg", "fixed_seg"):
        missing_labels = set(source_labels[name]) - set(output_labels[name]) - {0}
        if missing_labels:
            raise ValueError(
                "resampling removed labels %s for patient %s (%s)"
                % (sorted(missing_labels), case.patient_id, name)
            )

    arrays_dhw = {
        "moving": np.transpose(moving_xyz, (2, 1, 0)).astype(np.float32, copy=False),
        "fixed": np.transpose(fixed_xyz, (2, 1, 0)).astype(np.float32, copy=False),
        "moving_seg": np.transpose(moving_seg_xyz, (2, 1, 0)).astype(np.int16, copy=False),
        "fixed_seg": np.transpose(fixed_seg_xyz, (2, 1, 0)).astype(np.int16, copy=False),
    }
    expected_shape = tuple(int(value) for value in target_shape_dhw)
    for name, array in arrays_dhw.items():
        if array.shape != expected_shape:
            raise AssertionError("prepared %s shape mismatch for patient %s" % (name, case.patient_id))
        _save_npy(output_paths[name], array)

    target_linear = target_affine[:3, :3]
    target_direction = target_linear / np.linalg.norm(target_linear, axis=0)
    metadata: Dict[str, object] = {
        "case_id": output_id,
        "patient_id": case.patient_id,
        "pair_definition": "%s pre-RT moving to mid-RT fixed" % prealignment,
        "prealigned": True,
        "prealignment": prealignment_record,
        "array_axis_order": "zyx",
        "target_shape": list(expected_shape),
        "target_spacing": [float(value) for value in target_spacing_dhw],
        "target_affine_nifti_ras": target_affine.tolist(),
        "target_direction_nifti_ras": target_direction.tolist(),
        "target_origin_nifti_ras": target_affine[:3, 3].tolist(),
        "crop_frame": "fixed-image physical center",
        "label_map": {"background": 0, "GTVp": 1, "GTVn": 2},
        "source_paths": {
            "moving": str((case.pre_image if prealignment == "rigid-affine" else case.registered_pre_image).resolve()),
            "moving_seg": str(
                (case.pre_mask if prealignment == "rigid-affine" else case.registered_pre_mask).resolve()
            ),
            "fixed": str(case.mid_image.resolve()),
            "fixed_seg": str(case.mid_mask.resolve()),
            "official_registered_moving_qa": str(case.registered_pre_image.resolve()),
            "official_registered_moving_seg_qa": str(case.registered_pre_mask.resolve()),
        },
        "source_shape_xyz": list(fixed_image.shape),
        "source_spacing_xyz": [float(value) for value in fixed_image.header.get_zooms()[:3]],
        "source_affine_nifti_ras": np.asarray(fixed_image.affine, dtype=np.float64).tolist(),
        "source_orientation": list(nib.aff2axcodes(fixed_image.affine)),
        "normalization": {
            "moving": moving_normalization,
            "fixed": fixed_normalization,
        },
        "source_labels": source_labels,
        "output_labels": output_labels,
        "output_paths": {
            name: str(path.relative_to(output_root))
            for name, path in output_paths.items()
            if name != "metadata"
        },
    }
    output_paths["metadata"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def stratified_patient_split(
    records: Sequence[Mapping[str, object]],
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 1234,
) -> Dict[str, List[str]]:
    if not records:
        raise ValueError("at least one prepared patient record is required")
    if validation_fraction < 0.0 or test_fraction < 0.0:
        raise ValueError("split fractions must be non-negative")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation and test fractions must sum to less than one")

    groups: Dict[Tuple[int, ...], List[str]] = {}
    for record in records:
        labels = record["output_labels"]["fixed_seg"]  # type: ignore[index]
        signature = tuple(sorted(int(value) for value in labels if int(value) != 0))
        groups.setdefault(signature, []).append(str(record["case_id"]))

    rng = np.random.default_rng(int(seed))
    split = {"train": [], "val": [], "test": []}
    for signature in sorted(groups):
        patient_ids = sorted(groups[signature])
        rng.shuffle(patient_ids)
        count = len(patient_ids)
        if count >= 3:
            num_val = max(1, int(round(count * validation_fraction))) if validation_fraction > 0 else 0
            num_test = max(1, int(round(count * test_fraction))) if test_fraction > 0 else 0
            while num_val + num_test >= count:
                if num_test >= num_val and num_test > 0:
                    num_test -= 1
                elif num_val > 0:
                    num_val -= 1
        else:
            num_val = 0
            num_test = 0
        split["val"].extend(patient_ids[:num_val])
        split["test"].extend(patient_ids[num_val : num_val + num_test])
        split["train"].extend(patient_ids[num_val + num_test :])

    for name in split:
        split[name] = sorted(split[name])
    assigned = [patient_id for values in split.values() for patient_id in values]
    if len(assigned) != len(set(assigned)) or len(assigned) != len(records):
        raise AssertionError("patient-level split is not exhaustive and disjoint")
    return split


def _write_pair_manifest(
    path: Path,
    patient_ids: Sequence[str],
    records_by_id: Mapping[str, Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("moving", "fixed", "moving_seg", "fixed_seg", "patient_id"),
        )
        writer.writeheader()
        for patient_id in patient_ids:
            record = records_by_id[patient_id]
            output_paths = record["output_paths"]
            writer.writerow(
                {
                    "moving": output_paths["moving"],
                    "fixed": output_paths["fixed"],
                    "moving_seg": output_paths["moving_seg"],
                    "fixed_seg": output_paths["fixed_seg"],
                    "patient_id": patient_id,
                }
            )


def _preprocess_case_worker(
    case: HNTSMRG24Case,
    output_root: str,
    target_spacing_dhw: Tuple[float, float, float],
    target_shape_dhw: Tuple[int, int, int],
    lower_percentile: float,
    upper_percentile: float,
    prealignment: str,
    registration_seed: int,
    rigid_iterations: int,
    affine_iterations: int,
    overwrite: bool,
) -> Dict[str, object]:
    return preprocess_hntsmrg24_case(
        case,
        output_root=output_root,
        target_spacing_dhw=target_spacing_dhw,
        target_shape_dhw=target_shape_dhw,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        prealignment=prealignment,
        registration_seed=registration_seed,
        rigid_iterations=rigid_iterations,
        affine_iterations=affine_iterations,
        overwrite=overwrite,
    )


def prepare_hntsmrg24_dataset(
    source_root: str | Path,
    output_root: str | Path,
    manifest_dir: str | Path,
    target_spacing_dhw: Sequence[float] = (1.5, 1.5, 1.5),
    target_shape_dhw: Sequence[int] = (128, 160, 160),
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 1234,
    prealignment: str = "rigid-affine",
    rigid_iterations: int = 200,
    affine_iterations: int = 200,
    num_workers: int = 1,
    overwrite: bool = False,
) -> Dict[str, object]:
    cases = discover_hntsmrg24_cases(source_root)
    geometry_summary = inspect_hntsmrg24_geometry(cases)
    output_root = Path(output_root)
    manifest_dir = Path(manifest_dir)
    spacing = tuple(float(value) for value in target_spacing_dhw)
    shape = tuple(int(value) for value in target_shape_dhw)
    if len(spacing) != 3 or len(shape) != 3:
        raise ValueError("target spacing and shape must contain three values")
    if int(num_workers) < 1:
        raise ValueError("num_workers must be at least one")
    prealignment = str(prealignment).lower()
    if prealignment not in {"rigid-affine", "official-deformable"}:
        raise ValueError("prealignment must be 'rigid-affine' or 'official-deformable'")
    for directory in ("images", "seg_o", "metadata"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    if int(num_workers) == 1:
        for index, case in enumerate(cases, start=1):
            records.append(
                _preprocess_case_worker(
                    case,
                    str(output_root),
                    spacing,
                    shape,
                    float(lower_percentile),
                    float(upper_percentile),
                    prealignment,
                    int(seed) + int(case.patient_id),
                    int(rigid_iterations),
                    int(affine_iterations),
                    bool(overwrite),
                )
            )
            print("prepared HNTS-MRG patient %s (%d/%d)" % (case.patient_id, index, len(cases)))
    else:
        with ProcessPoolExecutor(max_workers=int(num_workers)) as executor:
            futures = {
                executor.submit(
                    _preprocess_case_worker,
                    case,
                    str(output_root),
                    spacing,
                    shape,
                    float(lower_percentile),
                    float(upper_percentile),
                    prealignment,
                    int(seed) + int(case.patient_id),
                    int(rigid_iterations),
                    int(affine_iterations),
                    bool(overwrite),
                ): case.patient_id
                for case in cases
            }
            for index, future in enumerate(as_completed(futures), start=1):
                patient_id = futures[future]
                records.append(future.result())
                print("prepared HNTS-MRG patient %s (%d/%d)" % (patient_id, index, len(cases)))

    records.sort(key=lambda record: int(str(record["patient_id"])))
    split = stratified_patient_split(
        records,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    records_by_id = {str(record["case_id"]): record for record in records}
    for name in ("train", "val", "test"):
        _write_pair_manifest(
            manifest_dir / (name + "_pairs.csv"),
            split[name],
            records_by_id,
        )
        (manifest_dir / (name + "_patients.txt")).write_text(
            "".join(patient_id + "\n" for patient_id in split[name]),
            encoding="utf-8",
        )

    signature_counts: Dict[str, int] = {}
    registration_selection_counts: Dict[str, int] = {}
    rigid_fallback_patients = []
    for record in records:
        labels = tuple(
            int(value) for value in record["output_labels"]["fixed_seg"] if int(value) != 0  # type: ignore[index]
        )
        key = "empty" if not labels else "+".join(str(value) for value in labels)
        signature_counts[key] = signature_counts.get(key, 0) + 1
        selected_stage = str(record["prealignment"].get("selected_stage", prealignment))  # type: ignore[union-attr]
        registration_selection_counts[selected_stage] = registration_selection_counts.get(selected_stage, 0) + 1
        if selected_stage == "rigid" and prealignment == "rigid-affine":
            rigid_fallback_patients.append(str(record["case_id"]))
    warnings = []
    if prealignment == "official-deformable":
        warnings.append(
            "official deformably registered pre-RT volumes are intended for QA only and can leak "
            "the target registration into a learned-registration benchmark"
        )
    if rigid_fallback_patients:
        warnings.append(
            "%d patients failed affine safety checks and used the accepted rigid transform; "
            "review prealignment.rigid_fallback_patients" % len(rigid_fallback_patients)
        )
    summary: Dict[str, object] = {
        "dataset": "HNTS-MRG 2024 training",
        "pair_definition": "%s pre-RT moving to mid-RT fixed" % prealignment,
        "source_root": str(Path(source_root).resolve()),
        "output_root": str(output_root.resolve()),
        "geometry_preflight": geometry_summary,
        "target_shape_dhw": list(shape),
        "target_spacing_dhw": list(spacing),
        "normalization": {
            "method": "per-volume robust percentile min-max",
            "lower_percentile": float(lower_percentile),
            "upper_percentile": float(upper_percentile),
        },
        "label_map": {"background": 0, "GTVp": 1, "GTVn": 2},
        "fixed_label_signature_counts": signature_counts,
        "split_seed": int(seed),
        "prealignment": {
            "mode": prealignment,
            "rigid_iterations": int(rigid_iterations) if prealignment == "rigid-affine" else 0,
            "affine_iterations": int(affine_iterations) if prealignment == "rigid-affine" else 0,
            "official_registered_volumes_used_for_qa_only": prealignment == "rigid-affine",
            "selected_stage_counts": registration_selection_counts,
            "rigid_fallback_patients": rigid_fallback_patients,
        },
        "split_counts": {name: len(values) for name, values in split.items()},
        "num_pairs": len(records),
        "warnings": warnings,
    }
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "model": {
            "use_anatomy_head": True,
            "num_anatomy_classes": 3,
        },
        "loss": {
            "jacobian_minimum_determinant": 0.1,
            "jacobian_tail_fraction": 0.0001,
            "jacobian_tail_weight": 1.0,
            "weights": {
                "dice": 1.0,
                "anatomy": 0.1,
                "jacobian": 5.0,
            }
        },
        "training": {
            "learning_rate": 2.0e-5,
            "stage_schedules": {
                "registration-warmup": {
                    "ramp_steps": 2000,
                    "anchor_start": 0.1,
                    "jacobian_start": 5.0,
                }
            },
        },
        "data": {
            "image_normalization": "zero_one",
            "target_shape": list(shape),
            "spacing_dhw": list(spacing),
        },
    }
    with (manifest_dir / "dataset_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return summary
