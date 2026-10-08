"""Read-only validation tests; fixture rows use the actual shared processor."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.dataset.dataset_utils import CSV_COLUMNS, DatasetWriter
from src.dataset.validator import validate_dataset
from src.hand_tracking.landmark_processor import create_feature_vector


class DatasetValidatorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "samples.csv"
        self.writer = DatasetWriter(self.path)

    def save_sample(self, number=0, label="HELLO", session="s1", reference="UNVERIFIED"):
        landmarks = [SimpleNamespace(x=0.3 + index * 0.01, y=0.4 + index * 0.005,
                                     z=-index * 0.002) for index in range(21)]
        landmarks[4].z += number * 0.01
        result = SimpleNamespace(hand_landmarks=[landmarks],
                                 handedness=[[SimpleNamespace(category_name="Left")]])
        self.writer.save(create_feature_vector(result, (640, 480)), label=label,
                         participant_id="p1", session_id=session, hand_count=1,
                         frame_size=(640, 480), source_reference=reference)

    def read_rows(self):
        with self.path.open(newline="", encoding="utf-8") as stream:
            return list(csv.reader(stream))

    def write_rows(self, rows):
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(rows)

    def test_valid_dataset_passes_checks_without_modification(self):
        for number, label, session in ((0, "HELLO", "s1"), (1, "HELLO", "s2"),
                                       (2, "HELP", "s3"), (3, "HELP", "s4")):
            self.save_sample(number, label, session, reference="Fixture reference, not real FSL evidence")
        original = self.path.read_bytes()
        metadata = self.writer.metadata_path.read_bytes()
        report = validate_dataset(self.path, min_samples=2)
        self.assertEqual(report.errors, [])
        self.assertEqual(report.valid_rows, 4)
        self.assertEqual(report.label_counts, {"HELLO": 2, "HELP": 2})
        self.assertTrue(report.passes_basic_training_checks)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(self.writer.metadata_path.read_bytes(), metadata)

    def test_practice_dataset_is_valid_but_not_training_ready(self):
        self.save_sample()
        report = validate_dataset(self.path)
        self.assertFalse(report.errors)
        self.assertFalse(report.passes_basic_training_checks)
        self.assertEqual(report.unverified_rows, 1)
        warnings = " ".join(report.warnings)
        for message in ("UNVERIFIED", "two populated classes", "one participant", "one session", "minimum"):
            self.assertIn(message, warnings)

    def test_empty_dataset_reports_no_samples(self):
        report = validate_dataset(self.path)
        self.assertIn("No valid unique samples", " ".join(report.errors))

    def test_missing_nonnumeric_and_nonfinite_features(self):
        self.save_sample()
        original = self.read_rows()
        feature_index = CSV_COLUMNS.index("f8")
        for invalid in ("", "not-a-number", "NaN", "inf"):
            rows = [row.copy() for row in original]
            rows[1][feature_index] = invalid
            self.write_rows(rows)
            report = validate_dataset(self.path)
            self.assertEqual(report.invalid_rows, 1)
            self.assertEqual(report.valid_rows, 0)

    def test_unknown_label_is_reported(self):
        self.save_sample()
        rows = self.read_rows()
        rows[1][0] = "INVENTED_LABEL"
        self.write_rows(rows)
        report = validate_dataset(self.path)
        self.assertIn("Unknown label", " ".join(report.errors))

    def test_duplicate_rows_features_and_conflicting_labels(self):
        self.save_sample()
        rows = self.read_rows()
        rows.append(rows[1].copy())
        conflicting = rows[1].copy()
        conflicting[0] = "HELP"
        rows.append(conflicting)
        self.write_rows(rows)
        report = validate_dataset(self.path)
        self.assertEqual(report.duplicate_rows, 1)
        self.assertEqual(report.duplicate_features, 2)
        self.assertEqual(report.conflicting_labels, 1)
        self.assertEqual(report.valid_rows, 1)
        self.assertEqual(report.invalid_rows, 2)

    def test_imbalanced_classes_are_reported(self):
        for number in range(4):
            self.save_sample(number, "HELLO", f"s{number}")
        self.save_sample(5, "HELP", "s5")
        report = validate_dataset(self.path, min_samples=1)
        self.assertIn("Class imbalance", " ".join(report.warnings))

    def test_metadata_mismatch_and_missing_metadata(self):
        self.save_sample()
        metadata = json.loads(self.writer.metadata_path.read_text())
        metadata["preprocessing_version"] = "other-version"
        self.writer.metadata_path.write_text(json.dumps(metadata))
        self.assertIn("metadata differs", " ".join(validate_dataset(self.path).errors))
        self.writer.metadata_path.unlink()
        self.assertIn("Missing/unreadable", " ".join(validate_dataset(self.path).errors))

    def test_incorrect_header_and_row_lengths(self):
        self.save_sample()
        original = self.read_rows()
        self.write_rows([original[0][:-1], original[1][:-1]])
        self.assertIn("Incorrect CSV header", " ".join(validate_dataset(self.path).errors))
        for invalid_row in (original[1][:-1], original[1] + ["extra"]):
            self.write_rows([original[0], invalid_row])
            report = validate_dataset(self.path)
            self.assertEqual(report.invalid_rows, 1)
            self.assertIn("Incorrect feature/row length", " ".join(report.errors))

    def test_invalid_normalization_and_hand_count(self):
        self.save_sample()
        original = self.read_rows()
        for column, value, message in (("f1", "0.5", "wrist"), ("hand_count", "2", "disagrees"),
                                       ("frame_width", "0", "positive integers"),
                                       ("captured_at_utc", "2026-10-07", "UTC offset")):
            rows = [row.copy() for row in original]
            rows[1][CSV_COLUMNS.index(column)] = value
            self.write_rows(rows)
            self.assertIn(message, " ".join(validate_dataset(self.path).errors))

    def test_session_cannot_identify_two_participants(self):
        self.save_sample(0)
        self.save_sample(1)
        rows = self.read_rows()
        rows[2][CSV_COLUMNS.index("participant_id")] = "p2"
        self.write_rows(rows)
        self.assertIn("different participants", " ".join(validate_dataset(self.path).errors))

    def test_cli_exit_statuses(self):
        from scripts import check_dataset
        self.save_sample()
        base_args = ["check_dataset.py", "--dataset", str(self.path)]
        with contextlib.redirect_stdout(io.StringIO()):
            with patch("sys.argv", base_args):
                self.assertEqual(check_dataset.main(), 0)
            with patch("sys.argv", base_args + ["--require-training-ready"]):
                self.assertEqual(check_dataset.main(), 2)
            rows = self.read_rows()
            rows[1][0] = "INVALID"
            self.write_rows(rows)
            with patch("sys.argv", base_args):
                self.assertEqual(check_dataset.main(), 1)


if __name__ == "__main__":
    unittest.main()
