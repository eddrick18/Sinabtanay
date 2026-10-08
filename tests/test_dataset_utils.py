"""Phase 4 persistence/policy checks using disposable numerical data only."""

import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from config import settings
from src.dataset.collector import SampleCollector
from src.dataset.dataset_utils import CSV_COLUMNS, DatasetWriter, FEATURE_COLUMNS


def hand_result(label="Left", variation=0.0):
    landmarks = [SimpleNamespace(x=0.3 + index * 0.01, y=0.4 + index * 0.005,
                                 z=-index * 0.002) for index in range(21)]
    landmarks[4].z += variation
    return SimpleNamespace(hand_landmarks=[landmarks],
                           handedness=[[SimpleNamespace(category_name=label)]])


class DatasetCollectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "landmarks.csv"
        self.writer = DatasetWriter(self.path)
        self.collector = SampleCollector(self.writer, "HELLO", "test_signer", "test_session",
                                         "UNVERIFIED", target=3)

    def rows(self):
        with self.path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_capture_saves_one_row_and_reload_restores_counts(self):
        self.assertEqual(self.collector.capture(hand_result(), (640, 480), 0), "Saved sample")
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(list(rows[0]), CSV_COLUMNS)
        self.assertEqual(rows[0]["label"], "HELLO")
        self.assertEqual(rows[0]["participant_id"], "test_signer")
        self.assertEqual(rows[0]["session_id"], "test_session")
        self.assertEqual(rows[0]["source_reference"], "UNVERIFIED")
        self.assertEqual(len(FEATURE_COLUMNS), 126)
        reloaded = DatasetWriter(self.path)
        self.assertEqual(reloaded.sample_counts["HELLO"], 1)
        self.assertEqual(reloaded.session_counts[("HELLO", "test_session")], 1)
        self.assertEqual(len(reloaded.seen_features), 1)

    def test_no_hand_and_ambiguous_hand_are_never_saved(self):
        empty = SimpleNamespace(hand_landmarks=[], handedness=[])
        self.assertIn("no hand", self.collector.capture(empty, (640, 480), 0))
        self.assertIn("Skipped", self.collector.capture(hand_result("Unknown"), (640, 480), 1))
        self.assertEqual(self.rows(), [])

    def test_duplicate_protection_survives_restart(self):
        self.collector.capture(hand_result(), (640, 480), 0)
        reloaded = DatasetWriter(self.path)
        collector = SampleCollector(reloaded, "HELLO", "test_signer", "new_session",
                                    "UNVERIFIED", 3)
        self.assertIn("duplicate", collector.capture(hand_result(), (640, 480), 0))
        other_label = SampleCollector(reloaded, "HELP", "test_signer", "other_session",
                                      "UNVERIFIED", 3)
        self.assertIn("different label", other_label.capture(hand_result(), (640, 480), 0))
        self.assertEqual(len(self.rows()), 1)

    def test_cooldown_and_target_prevent_extra_rows(self):
        self.collector.target = 2
        self.collector.capture(hand_result(), (640, 480), 0)
        self.assertIn("cooldown", self.collector.capture(hand_result(variation=0.01), (640, 480), 0.1))
        self.assertEqual(self.collector.capture(hand_result(variation=0.01), (640, 480), 1), "Saved sample")
        self.assertIn("target reached", self.collector.capture(hand_result(variation=0.02), (640, 480), 2))
        self.assertEqual(len(self.rows()), 2)

    def test_incompatible_schema_refuses_append(self):
        metadata = json.loads(self.writer.metadata_path.read_text())
        metadata["mirror_camera"] = not metadata["mirror_camera"]
        self.writer.metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            DatasetWriter(self.path)

    def test_broken_existing_row_refuses_append(self):
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(["HELLO", "truncated"])
        with self.assertRaisesRegex(ValueError, "row 2"):
            DatasetWriter(self.path)

    def test_direct_writer_rejects_bad_features_and_unknown_label(self):
        context = dict(label="HELLO", participant_id="test_signer", session_id="test_session",
                       hand_count=1, frame_size=(640, 480), source_reference="UNVERIFIED")
        for features in (np.zeros(126), np.ones(63), np.full(126, np.nan)):
            with self.assertRaises(ValueError):
                self.writer.save(features, **context)
        context["label"] = "NOT_CONFIGURED"
        with self.assertRaises(ValueError):
            self.writer.save(np.ones(126), **context)
        self.assertEqual(self.rows(), [])

    def test_existing_csv_without_metadata_refuses_append(self):
        self.writer.metadata_path.unlink()
        with self.assertRaisesRegex(ValueError, "no schema metadata"):
            DatasetWriter(self.path)

    def test_webcam_loop_only_captures_on_space_and_cleans_up(self):
        # Exercise the real orchestration with simulated camera/keys, not real hardware.
        from scripts import collect_data
        empty = SimpleNamespace(hand_landmarks=[], handedness=[])
        camera = MagicMock()
        camera.isOpened.return_value = True
        camera.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        detector = MagicMock()
        detector.detect.side_effect = [empty, hand_result(), hand_result(), hand_result()]
        arguments = SimpleNamespace(dataset=self.path, label="HELLO", participant_id="test_signer",
                                    reference="UNVERIFIED", target=3, camera_index=0, debug=False)
        with patch.object(collect_data, "HandDetector", return_value=detector), \
             patch.object(collect_data.cv2, "VideoCapture", return_value=camera), \
             patch.object(collect_data.cv2, "imshow"), \
             patch.object(collect_data.cv2, "waitKey", side_effect=[32, ord("s"), 32, ord("q")]), \
             patch.object(collect_data.cv2, "getWindowProperty", return_value=1), \
             patch.object(collect_data.cv2, "destroyAllWindows") as destroy_windows, \
             patch.object(collect_data.time, "monotonic", side_effect=[0.0, 1.0]):
            self.assertEqual(collect_data.run_collection(arguments), 0)
        self.assertEqual(len(self.rows()), 1)
        camera.release.assert_called_once()
        detector.close.assert_called_once()
        destroy_windows.assert_called_once()


if __name__ == "__main__":
    unittest.main()
