import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gam_reg.data.dataset_layout import (
    build_training_pairs,
    prepare_layout_manifests,
    read_id_pairs,
)
from gam_reg.config import load_config


class DatasetLayoutTest(unittest.TestCase):
    def _write_case(self, root: Path, case_id: str, label: int) -> None:
        image = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(4, 4, 4)
        segmentation = np.zeros((4, 4, 4), dtype=np.int16)
        segmentation[1:3, 1:3, 1:3] = label
        np.save(root / "images" / (case_id + ".npy"), image)
        np.save(root / "seg_o" / (case_id + ".npy"), segmentation)
        metadata = {
            "case_id": case_id,
            "target_shape": [4, 4, 4],
            "target_spacing": [2.0, 2.0, 2.0],
            "label_map": {"background": 0, "organ": 1},
            "array_axis_order": "zyx",
            "target_direction": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "target_origin": [0.0, 0.0, 0.0],
            "prealigned": True,
        }
        (root / "metadata" / (case_id + ".json")).write_text(json.dumps(metadata), encoding="utf-8")

    def _make_layout(self, root: Path) -> None:
        for directory in ("images", "seg_o", "metadata", "lists/paper_split"):
            (root / directory).mkdir(parents=True, exist_ok=True)
        for case_id in ("case_a", "case_b", "case_c", "case_d", "case_e", "case_f"):
            self._write_case(root, case_id, label=1)
        split = root / "lists" / "paper_split"
        (split / "trn_list_inter.txt").write_text("case_a\ncase_b\n", encoding="utf-8")
        (split / "val_pairs.csv").write_text("case_c,case_d\n", encoding="utf-8")
        (split / "test_pairs.csv").write_text("moving_id,fixed_id\ncase_e,case_f\n", encoding="utf-8")

    def test_training_pairs_exclude_self_by_default(self):
        self.assertEqual(build_training_pairs(["a", "b"]), [("a", "b"), ("b", "a")])
        self.assertEqual(len(build_training_pairs(["a", "b"], include_self=True)), 4)

    def test_read_pair_ids_accepts_headerless_and_headered_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            headerless = root / "headerless.csv"
            headered = root / "headered.csv"
            headerless.write_text("a,b\n", encoding="utf-8")
            headered.write_text("moving,fixed\na,b\n", encoding="utf-8")
            self.assertEqual(read_id_pairs(headerless), [("a", "b")])
            self.assertEqual(read_id_pairs(headered), [("a", "b")])

    def test_prepare_layout_validates_and_writes_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            output = Path(tmp) / "manifests"
            self._make_layout(root)
            summary = prepare_layout_manifests(root, output, expected_shape=(4, 4, 4))
            self.assertEqual(summary["num_cases"], 6)
            self.assertEqual(summary["num_anatomy_classes"], 2)
            self.assertEqual(summary["manifest_counts"]["train_pairs"], 2)
            self.assertEqual(summary["warnings"], [])
            with (output / "train_pairs.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["moving"], "images/case_a.npy")
            self.assertEqual(rows[0]["moving_seg"], "seg_o/case_a.npy")
            self.assertTrue((output / "dataset_summary.json").is_file())
            config = load_config(output / "dataset_config.yaml")
            self.assertEqual(config["model"]["num_anatomy_classes"], 2)
            self.assertEqual(config["data"]["image_normalization"], "zero_one")
            self.assertEqual(config["data"]["target_shape"], [4, 4, 4])

    def test_prepare_layout_rejects_out_of_range_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            self._make_layout(root)
            np.save(root / "images" / "case_a.npy", np.full((4, 4, 4), 2.0, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
                prepare_layout_manifests(root, Path(tmp) / "manifests")

    def test_prepare_layout_rejects_case_level_split_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            self._make_layout(root)
            split = root / "lists" / "paper_split"
            (split / "val_pairs.csv").write_text("case_a,case_c\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "split leakage"):
                prepare_layout_manifests(root, Path(tmp) / "manifests")


if __name__ == "__main__":
    unittest.main()
