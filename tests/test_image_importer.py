"""Import rules use small synthetic images and mocked landmarks, not FSL evidence."""

import csv
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from src.dataset.image_importer import build_source_groups, import_images, merge_identical_feature_families
from src.dataset.validator import validate_dataset
from src.training.trainer import load_training_data, split_dataset, train_model


def fake_detection(frame):
    intensity = float(frame.mean())
    if intensity < 1:
        return SimpleNamespace(hand_landmarks=[], handedness=[])
    landmarks = [SimpleNamespace(x=0.3 + i * 0.01, y=0.4 + i * 0.005, z=-i * 0.002) for i in range(21)]
    landmarks[4].z += intensity * 0.001
    return SimpleNamespace(hand_landmarks=[landmarks],
                           handedness=[[SimpleNamespace(category_name="Left")]])


class ImageImporterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "Collated"
        self.output = self.root / "alphabet.csv"
        self.hand_model = self.root / "mock_hand_landmarker.task"
        self.hand_model.write_bytes(b"mock model for provenance hashing")
        self.detector = MagicMock()
        self.detector.detect.side_effect = fake_detection

    def image(self, label, filename, intensity):
        path = self.source / label / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.full((100, 120, 3), intensity, dtype=np.uint8))
        return path

    def run_import(self):
        with patch("src.dataset.image_importer.HandDetector", return_value=self.detector), \
                patch("src.dataset.image_importer.settings.HAND_MODEL_PATH", self.hand_model):
            return import_images(self.source, self.output)

    def test_import_schema_validation_and_group_split_preserve_unknowns(self):
        for label, filename, intensity in (("A", "1.jpg", 10), ("A", "1_jpg.rf.abc.jpg", 20),
                                           ("A", "2.jpg", 30), ("B", "3.jpg", 40), ("B", "4.jpg", 50)):
            self.image(label, filename, intensity)
        original = {path: path.read_bytes() for path in self.source.rglob("*.jpg")}
        summary = self.run_import()
        self.assertEqual(summary["retained_samples"], 5)
        self.assertEqual(summary["retained_source_groups"], 4)
        report = validate_dataset(self.output, min_samples=1)
        self.assertFalse(report.errors)
        self.assertEqual(report.label_counts, {"A": 3, "B": 2})
        self.assertEqual(report.unverified_rows, 5)
        self.assertEqual(len(report.participants), 0)
        frame, provenance = load_training_data(self.output, practice=True)
        self.assertTrue((frame.participant_id == "UNKNOWN").all())
        self.assertTrue((frame.session_id == "UNKNOWN").all())
        self.assertTrue((frame.captured_at_utc == "UNKNOWN").all())
        train, test = split_dataset(frame, "group", 0.2, 42)
        self.assertFalse(set(frame.iloc[train].source_group_id) & set(frame.iloc[test].source_group_id))
        with self.assertRaisesRegex(ValueError, "Unknown recording sessions"):
            split_dataset(frame, "session", 0.2, 42)
        for path, content in original.items():
            self.assertEqual(path.read_bytes(), content)
        self.detector.close.assert_called_once()
        with contextlib.redirect_stdout(io.StringIO()):
            metadata = train_model(self.output, self.root / "test_model", practice=True, tree_count=8)
        self.assertEqual(metadata["split_mode"], "group")
        self.assertEqual(metadata["source_group_overlap"], 0)
        self.assertIsNone(metadata["session_overlap"])
        self.assertIsNone(metadata["participant_overlap"])

    def test_skips_duplicates_dynamic_letters_unreadable_and_no_hand(self):
        first = self.image("A", "1.jpg", 10)
        duplicate = self.image("A", "duplicate.jpg", 20)
        duplicate.write_bytes(first.read_bytes())
        self.image("B", "3.jpg", 30)
        self.image("J", "4.jpg", 40)
        self.image("Z", "5.jpg", 50)
        self.image("A", "no_hand.jpg", 0)
        broken = self.source / "A" / "broken.jpg"
        broken.write_bytes(b"not an image")
        report = self.run_import()
        self.assertEqual(report["retained_samples"], 2)
        self.assertEqual(report["status_counts"]["duplicate_image"], 1)
        self.assertEqual(report["status_counts"]["excluded_dynamic_letter"], 2)
        self.assertEqual(report["status_counts"]["no_hand"], 1)
        self.assertEqual(report["status_counts"]["unreadable_image"], 1)
        with self.output.with_suffix(".audit.csv").open(newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 7)

    def test_conflicting_source_labels_withhold_both_sides(self):
        a = self.image("A", "1.jpg", 10)
        b = self.image("B", "2.jpg", 20)
        b.write_bytes(a.read_bytes())
        self.image("C", "3.jpg", 30)
        report = self.run_import()
        self.assertEqual(report["status_counts"]["conflicting_source_labels"], 2)
        self.assertEqual(report["retained_class_counts"], {"C": 1})

    def test_conflicting_feature_labels_withhold_both_sides(self):
        self.image("A", "1.jpg", 10)
        self.image("B", "2.jpg", 20)
        self.detector.detect.side_effect = None
        self.detector.detect.return_value = fake_detection(np.full((10, 10, 3), 10))
        report = self.run_import()
        self.assertEqual(report["retained_samples"], 0)
        self.assertEqual(report["status_counts"]["conflicting_feature_labels"], 2)

    def test_same_label_feature_duplicates_are_not_saved_twice(self):
        self.image("A", "1.jpg", 10)
        self.image("A", "2.jpg", 20)
        self.detector.detect.side_effect = None
        self.detector.detect.return_value = fake_detection(np.full((10, 10, 3), 10))
        report = self.run_import()
        self.assertEqual(report["retained_samples"], 1)
        self.assertEqual(report["status_counts"]["duplicate_features"], 1)

    def test_transitive_name_and_content_groups_are_connected(self):
        entries = [{"path": Path("A/1.jpg"), "source_file": "A/1.jpg", "label": "A", "source_sha256": "hash1"},
                   {"path": Path("A/1_jpg.rf.x.jpg"), "source_file": "A/1_jpg.rf.x.jpg", "label": "A", "source_sha256": "hash2"},
                   {"path": Path("A/2.jpg"), "source_file": "A/2.jpg", "label": "A", "source_sha256": "hash2"}]
        build_source_groups(entries)
        self.assertEqual(len({entry["source_group_id"] for entry in entries}), 1)

    def test_identical_features_link_whole_name_families(self):
        entries = [{"source_group_id": "group_a", "label": "A", "status": "candidate"},
                   {"source_group_id": "group_b", "label": "A", "status": "candidate"},
                   {"source_group_id": "group_b", "label": "A", "status": "candidate"}]
        candidates = [{"entry": entries[0], "key": (1,)}, {"entry": entries[1], "key": (1,)},
                      {"entry": entries[2], "key": (2,)}]
        merge_identical_feature_families(entries, candidates)
        self.assertEqual(len({entry["source_group_id"] for entry in entries}), 1)

    def test_existing_outputs_are_preserved(self):
        self.image("A", "1.jpg", 10)
        self.output.write_text("existing dataset")
        with self.assertRaisesRegex(ValueError, "already exist"):
            self.run_import()
        self.assertEqual(self.output.read_text(), "existing dataset")
