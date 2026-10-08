"""Camera-free checks: real no-hand inference and visualization edge cases."""

from config import settings
from types import SimpleNamespace
import unittest

import numpy as np

from src.hand_tracking.hand_detector import HandDetector, draw_hand_results, get_hand_labels


class HandDetectorTests(unittest.TestCase):
    @unittest.skipUnless(settings.HAND_MODEL_PATH.is_file(), "Download the MediaPipe hand model to run inference integration checks.")
    def test_blank_video_frames_have_no_hands(self):
        detector = HandDetector()
        try:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            for timestamp in (0, 0, 1):
                result = detector.detect(frame, timestamp)
                self.assertEqual(result.hand_landmarks, [])
                self.assertEqual(get_hand_labels(result), [])
        finally:
            detector.close()

    def test_two_hands_are_drawn_and_labeled(self):
        # Synthetic coordinates verify rendering, not actual hand-detection accuracy.
        hands = []
        for offset in (0.1, 0.6):
            hands.append([SimpleNamespace(x=offset + index * 0.005, y=0.5)
                          for index in range(21)])
        result = SimpleNamespace(
            hand_landmarks=hands,
            handedness=[[SimpleNamespace(category_name="Left")],
                        [SimpleNamespace(category_name="Right")]],
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_hand_results(frame, result)
        self.assertEqual(get_hand_labels(result), ["Left", "Right"])
        self.assertTrue(frame[:, :320].any())
        self.assertTrue(frame[:, 320:].any())

    def test_missing_handedness_has_a_fallback(self):
        result = SimpleNamespace(hand_landmarks=[[object()]], handedness=[])
        self.assertEqual(get_hand_labels(result), ["Unknown"])


if __name__ == "__main__":
    unittest.main()
