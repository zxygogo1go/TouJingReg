from __future__ import annotations

import tempfile
import unittest
import csv
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from gam_reg.data.hntsmrg24 import (
    discover_hntsmrg24_cases,
    inspect_hntsmrg24_geometry,
    prepare_hntsmrg24_dataset,
    preprocess_hntsmrg24_case,
    rigid_affine_prealign_case,
    robust_normalize_mri,
    stratified_patient_split,
)


def write_nifti(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


class HNTSMRG24PreprocessingTest(unittest.TestCase):
    def make_case(self, root: Path, patient_id: str, affine: np.ndarray | None = None) -> None:
        if affine is None:
            affine = np.diag([-1.5, -1.5, 2.0, 1.0])
        xx, yy, zz = np.meshgrid(
            np.linspace(-1.0, 1.0, 12),
            np.linspace(-1.0, 1.0, 14),
            np.linspace(-1.0, 1.0, 10),
            indexing="ij",
        )
        image = (
            100.0 * np.exp(-((xx + 0.25) ** 2 + (yy - 0.15) ** 2 + zz**2) / 0.12)
            + 60.0 * np.exp(-((xx - 0.35) ** 2 + (yy + 0.4) ** 2 + (zz - 0.2) ** 2) / 0.08)
        ).astype(np.float32)
        mask = np.zeros_like(image, dtype=np.int16)
        mask[4:7, 5:8, 3:6] = 1
        paths = {
            root / patient_id / "preRT" / (patient_id + "_preRT_T2.nii.gz"): image,
            root / patient_id / "preRT" / (patient_id + "_preRT_mask.nii.gz"): mask,
            root / patient_id / "midRT" / (patient_id + "_midRT_T2.nii.gz"): image + 1.0,
            root / patient_id / "midRT" / (patient_id + "_midRT_mask.nii.gz"): mask,
            root
            / patient_id
            / "midRT"
            / (patient_id + "_preRT_T2_registered.nii.gz"): image + 2.0,
            root
            / patient_id
            / "midRT"
            / (patient_id + "_preRT_mask_registered.nii.gz"): mask,
        }
        for path, data in paths.items():
            write_nifti(path, data, affine)

    def test_discovery_and_geometry_inspection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_case(root, "10")
            self.make_case(root, "2")
            cases = discover_hntsmrg24_cases(root)
            self.assertEqual([case.patient_id for case in cases], ["2", "10"])
            summary = inspect_hntsmrg24_geometry(cases)
            self.assertEqual(summary["num_patients"], 2)
            self.assertEqual(summary["orientation"], ["L", "P", "S"])
            self.assertEqual(summary["shape_xyz"]["minimum"], [12, 14, 10])

    def test_missing_required_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_case(root, "2")
            (root / "2" / "midRT" / "2_preRT_mask_registered.nii.gz").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "required HNTS-MRG file"):
                discover_hntsmrg24_cases(root)

    def test_registered_geometry_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_case(root, "2")
            path = root / "2" / "midRT" / "2_preRT_T2_registered.nii.gz"
            image = nib.load(str(path))
            affine = image.affine.copy()
            affine[0, 3] += 10.0
            write_nifti(path, np.asanyarray(image.dataobj), affine)
            cases = discover_hntsmrg24_cases(root)
            with self.assertRaisesRegex(ValueError, "affine mismatch"):
                inspect_hntsmrg24_geometry(cases)

    def test_pre_image_mask_geometry_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_case(root, "2")
            path = root / "2" / "preRT" / "2_preRT_mask.nii.gz"
            image = nib.load(str(path))
            affine = image.affine.copy()
            affine[1, 3] += 2.0
            write_nifti(path, np.asanyarray(image.dataobj), affine)
            cases = discover_hntsmrg24_cases(root)
            with self.assertRaisesRegex(ValueError, "pre-RT image/mask affine mismatch"):
                inspect_hntsmrg24_geometry(cases)

    def test_robust_mri_normalization_outputs_zero_one(self):
        volume = np.linspace(-10.0, 100.0, 1000, dtype=np.float32).reshape(10, 10, 10)
        normalized, record = robust_normalize_mri(volume)
        self.assertEqual(normalized.dtype, np.float32)
        self.assertGreaterEqual(float(normalized.min()), 0.0)
        self.assertLessEqual(float(normalized.max()), 1.0)
        self.assertLess(record["lower_value"], record["upper_value"])

    def test_case_preprocessing_preserves_geometry_labels_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            output = Path(tmp) / "prepared"
            self.make_case(root, "2")
            case = discover_hntsmrg24_cases(root)[0]
            metadata = preprocess_hntsmrg24_case(
                case,
                output,
                target_spacing_dhw=(1.5, 1.5, 1.5),
                target_shape_dhw=(12, 12, 12),
                prealignment="official-deformable",
            )
            moving = np.load(output / "images" / "hntsmrg24_002_pre_aligned.npy")
            fixed = np.load(output / "images" / "hntsmrg24_002_mid.npy")
            moving_seg = np.load(output / "seg_o" / "hntsmrg24_002_pre_aligned.npy")
            self.assertEqual(moving.shape, (12, 12, 12))
            self.assertEqual(fixed.shape, (12, 12, 12))
            self.assertEqual(moving.dtype, np.float32)
            self.assertEqual(moving_seg.dtype, np.int16)
            self.assertEqual(set(np.unique(moving_seg).tolist()), {0, 1})
            self.assertGreaterEqual(float(moving.min()), 0.0)
            self.assertLessEqual(float(moving.max()), 1.0)
            self.assertTrue(metadata["prealigned"])
            self.assertEqual(metadata["target_shape"], [12, 12, 12])
            self.assertTrue((output / "metadata" / "hntsmrg24_002.json").is_file())

    def test_stratified_split_is_disjoint_and_reproducible(self):
        records = []
        for index in range(12):
            labels = [0] if index < 4 else [0, 1, 2]
            records.append(
                {
                    "case_id": "hntsmrg24_%03d" % index,
                    "output_labels": {"fixed_seg": labels},
                }
            )
        first = stratified_patient_split(records, seed=7)
        second = stratified_patient_split(records, seed=7)
        self.assertEqual(first, second)
        assigned = [case_id for values in first.values() for case_id in values]
        self.assertEqual(len(assigned), 12)
        self.assertEqual(len(set(assigned)), 12)
        for split_name in ("val", "test"):
            signatures = {
                tuple(records[int(case_id[-3:])]["output_labels"]["fixed_seg"])
                for case_id in first[split_name]
            }
            self.assertEqual(signatures, {(0,), (0, 1, 2)})

    def test_dataset_preparation_writes_patient_level_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            output = Path(tmp) / "prepared"
            manifests = Path(tmp) / "manifests"
            for patient_id in range(1, 7):
                self.make_case(root, str(patient_id))
            summary = prepare_hntsmrg24_dataset(
                source_root=root,
                output_root=output,
                manifest_dir=manifests,
                target_spacing_dhw=(1.5, 1.5, 1.5),
                target_shape_dhw=(12, 12, 12),
                prealignment="official-deformable",
                num_workers=1,
            )
            self.assertEqual(summary["num_pairs"], 6)
            self.assertEqual(sum(summary["split_counts"].values()), 6)
            self.assertTrue(summary["warnings"])
            rows = []
            for name in ("train", "val", "test"):
                with (manifests / (name + "_pairs.csv")).open(newline="", encoding="utf-8") as handle:
                    rows.extend(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            self.assertEqual(len({row["patient_id"] for row in rows}), 6)
            self.assertTrue(all("pre_aligned" in row["moving"] for row in rows))
            config_text = (manifests / "dataset_config.yaml").read_text(encoding="utf-8")
            self.assertIn("num_anatomy_classes: 3", config_text)
            saved_summary = json.loads((manifests / "dataset_summary.json").read_text())
            self.assertEqual(saved_summary["pair_definition"], summary["pair_definition"])

    @unittest.skipUnless(importlib.util.find_spec("SimpleITK"), "SimpleITK is not installed")
    def test_rigid_affine_prealignment_is_finite_and_preserves_mask_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_case(root, "2")
            pre_image_path = root / "2" / "preRT" / "2_preRT_T2.nii.gz"
            pre_mask_path = root / "2" / "preRT" / "2_preRT_mask.nii.gz"
            pre_image = nib.load(str(pre_image_path))
            pre_mask = nib.load(str(pre_mask_path))
            shifted_image = np.zeros(pre_image.shape, dtype=np.float32)
            shifted_mask = np.zeros(pre_mask.shape, dtype=np.int16)
            shifted_image[2:] = np.asanyarray(pre_image.dataobj)[:-2]
            shifted_mask[2:] = np.asanyarray(pre_mask.dataobj)[:-2]
            write_nifti(pre_image_path, shifted_image, pre_image.affine)
            write_nifti(pre_mask_path, shifted_mask, pre_mask.affine)
            case = discover_hntsmrg24_cases(root)[0]
            moving, moving_mask, record = rigid_affine_prealign_case(
                case,
                seed=7,
                rigid_iterations=30,
                affine_iterations=30,
            )
            self.assertEqual(moving.shape, (12, 14, 10))
            self.assertEqual(moving_mask.shape, moving.shape)
            self.assertGreater(record["affine_determinant"], 0.5)
            self.assertLess(record["affine_determinant"], 2.0)
            self.assertTrue(np.isfinite(record["rigid_final_metric"]))
            self.assertTrue(np.isfinite(record["affine_final_metric"]))
            aligned_mask = np.asanyarray(moving_mask.dataobj)
            fixed_mask = np.asanyarray(nib.load(str(case.mid_mask)).dataobj)
            self.assertEqual(set(np.unique(aligned_mask).tolist()), {0, 1})
            before_distance = np.linalg.norm(
                np.argwhere(shifted_mask > 0).mean(axis=0)
                - np.argwhere(fixed_mask > 0).mean(axis=0)
            )
            after_distance = np.linalg.norm(
                np.argwhere(aligned_mask > 0).mean(axis=0)
                - np.argwhere(fixed_mask > 0).mean(axis=0)
            )
            self.assertLess(after_distance, before_distance)


if __name__ == "__main__":
    unittest.main()
