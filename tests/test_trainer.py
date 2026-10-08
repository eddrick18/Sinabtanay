"""Synthetic examples test engineering only; they are not verified FSL signs."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from src.dataset.dataset_utils import DatasetWriter, FEATURE_COLUMNS
from src.hand_tracking.landmark_processor import create_feature_vector
from src.training.evaluator import evaluate_predictions
from src.training.trainer import load_training_data, split_dataset, train_model


def build_synthetic_dataset(path: Path, sessions_per_class=5, samples_per_session=4):
    """Two arbitrary geometric classes, not a proposal for actual FSL gestures."""
    writer = DatasetWriter(path)
    random = np.random.default_rng(123)
    for label, direction in (("HELLO", 1.0), ("HELP", -1.0)):
        for session in range(sessions_per_class):
            for example in range(samples_per_session):
                coordinates = np.zeros((21, 3))
                coordinates[:, 0] = np.linspace(0, 0.2 * direction, 21)
                coordinates[:, 1] = np.linspace(0, 0.3, 21)
                coordinates[:, 2] = random.normal(0, 0.01, 21)
                landmarks = [SimpleNamespace(x=x, y=y, z=z) for x, y, z in coordinates]
                result = SimpleNamespace(hand_landmarks=[landmarks],
                                         handedness=[[SimpleNamespace(category_name="Left")]])
                writer.save(create_feature_vector(result, (640, 480)), label=label,
                            participant_id="synthetic", session_id=f"{label}_s{session}",
                            hand_count=1, frame_size=(640, 480), source_reference="UNVERIFIED")


class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.dataset = self.root / "synthetic.csv"
        build_synthetic_dataset(self.dataset)

    def test_session_split_is_disjoint_stratified_and_reproducible(self):
        frame, provenance = load_training_data(self.dataset, practice=True)
        train, test = split_dataset(frame, "session", 0.2, 42)
        self.assertFalse(set(train) & set(test))
        self.assertEqual(set(train) | set(test), set(range(len(frame))))
        self.assertFalse(set(frame.iloc[train].session_id) & set(frame.iloc[test].session_id))
        self.assertEqual(set(frame.iloc[train].label), {"HELLO", "HELP"})
        self.assertEqual(set(frame.iloc[test].label), {"HELLO", "HELP"})
        self.assertEqual(len(test), 8)
        repeated_train, repeated_test = split_dataset(frame, "session", 0.2, 42)
        np.testing.assert_array_equal(train, repeated_train)
        np.testing.assert_array_equal(test, repeated_test)

    def test_frame_split_is_disjoint_and_keeps_both_classes(self):
        frame, _ = load_training_data(self.dataset, practice=True)
        train, test = split_dataset(frame, "frame", 0.2, 42)
        self.assertFalse(set(train) & set(test))
        self.assertEqual(len(test), 8)
        self.assertEqual(set(frame.iloc[test].label), {"HELLO", "HELP"})

    def test_fit_save_and_reload_preserve_feature_contract_and_test_metrics(self):
        output = self.root / "run"
        original = self.dataset.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            metadata = train_model(self.dataset, output, practice=True, tree_count=8)
        self.assertTrue(metadata["practice"])
        self.assertEqual(metadata["feature_count"], 126)
        self.assertEqual(metadata["training_samples"], 32)
        self.assertEqual(metadata["testing_samples"], 8)
        self.assertEqual(metadata["session_overlap"], 0)
        self.assertEqual(self.dataset.read_bytes(), original)
        model = joblib.load(output / "trained/sign_classifier.joblib")
        self.assertEqual(model.n_features_in_, 126)
        frame, _ = load_training_data(self.dataset, practice=True)
        test = np.array(metadata["testing_csv_records"]) - 2
        features = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        actual_evaluation = evaluate_predictions(frame.iloc[test].label.to_numpy(),
                                                 model.predict(features[test]), metadata["labels"])
        saved_evaluation = json.loads((output / "metadata/evaluation.json").read_text())
        self.assertEqual(saved_evaluation["confusion_matrix"], actual_evaluation["confusion_matrix"])
        self.assertEqual(np.array(saved_evaluation["confusion_matrix"]).sum(), len(test))
        self.assertTrue((output / "metadata/confusion_matrix.png").read_bytes().startswith(b"\x89PNG"))
        for label in metadata["labels"]:
            for metric in ("precision", "recall", "f1-score", "support"):
                self.assertIn(metric, saved_evaluation["classification_report"][label])
        self.assertEqual(json.loads((output / "metadata/model_metadata.json").read_text()), metadata)

    def test_unverified_normal_run_is_refused_without_outputs(self):
        output = self.root / "refused"
        with self.assertRaisesRegex(ValueError, "UNVERIFIED"):
            train_model(self.dataset, output, tree_count=8)
        self.assertFalse(output.exists())

    def test_one_class_is_refused_even_in_practice(self):
        frame, _ = load_training_data(self.dataset, practice=True)
        frame = frame[frame.label == "HELLO"]
        frame.to_csv(self.dataset, index=False)
        with self.assertRaisesRegex(ValueError, "two populated labels"):
            load_training_data(self.dataset, practice=True)

    def test_insufficient_or_mixed_label_sessions_are_refused(self):
        frame, _ = load_training_data(self.dataset, practice=True)
        frame.loc[frame.label == "HELLO", "session_id"] = "one_hello_session"
        with self.assertRaisesRegex(ValueError, "two separate sessions"):
            split_dataset(frame, "session", 0.2, 42)
        frame["session_id"] = "one_mixed_session"
        with self.assertRaisesRegex(ValueError, "one label per session"):
            split_dataset(frame, "session", 0.2, 42)

    def test_small_session_split_adjusts_fraction_to_include_classes(self):
        frame, _ = load_training_data(self.dataset, practice=True)
        frame = frame[frame.session_id.str.endswith(("s0", "s1"))].reset_index(drop=True)
        train, test = split_dataset(frame, "session", 0.2, 42)
        self.assertEqual(len(train), len(test))
        self.assertEqual(set(frame.iloc[test].label), {"HELLO", "HELP"})

    def test_existing_model_is_preserved(self):
        output = self.root / "run"
        model_path = output / "trained/sign_classifier.joblib"
        model_path.parent.mkdir(parents=True)
        model_path.write_bytes(b"existing-model")
        with self.assertRaisesRegex(ValueError, "already exist"):
            train_model(self.dataset, output, practice=True, tree_count=8)
        self.assertEqual(model_path.read_bytes(), b"existing-model")

    def test_metrics_expose_wrong_predictions(self):
        evaluation = evaluate_predictions(["HELLO", "HELLO", "HELP", "HELP"],
                                          ["HELLO", "HELP", "HELP", "HELP"], ["HELLO", "HELP"])
        self.assertEqual(evaluation["accuracy"], 0.75)
        self.assertEqual(evaluation["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(evaluation["classification_report"]["HELLO"]["recall"], 0.5)

    def test_cli_pipeline_runs_and_invalid_dataset_has_clear_error(self):
        from scripts import train_model as command
        output = self.root / "cli-run"
        arguments = ["train_model.py", "--dataset", str(self.dataset), "--practice",
                     "--output-dir", str(output), "--trees", "8"]
        with patch("sys.argv", arguments), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(command.main(), 0)
        with patch("sys.argv", ["train_model.py", "--dataset", str(self.root / "missing.csv")]):
            self.assertEqual(command.main(), 1)


if __name__ == "__main__":
    unittest.main()
