"""Numerical preprocessing checks; no webcam or MediaPipe import required."""

from types import SimpleNamespace
import unittest

import numpy as np

from src.hand_tracking.landmark_processor import (
    create_feature_vector, extract_hand_landmarks, flatten_landmarks, normalize_landmarks,
)


def make_coordinates():
    coordinates = np.zeros((21, 3), dtype=np.float64)
    coordinates[:, 0] = np.linspace(0, 0.2, 21)
    coordinates[:, 1] = np.linspace(0, 0.3, 21)
    coordinates[:, 2] = np.linspace(0, -0.1, 21)
    return coordinates + [0.4, 0.5, 0.01]


def make_result(labels, coordinates=None):
    coordinates = make_coordinates() if coordinates is None else coordinates
    hands = [[SimpleNamespace(x=x, y=y, z=z) for x, y, z in coordinates]
             for label in labels]
    categories = [[SimpleNamespace(category_name=label)] for label in labels]
    return SimpleNamespace(hand_landmarks=hands, handedness=categories)


class LandmarkProcessorTests(unittest.TestCase):
    def test_valid_hand_extraction_and_flatten_order(self):
        expected = make_coordinates()
        actual = extract_hand_landmarks(make_result(["Left"]).hand_landmarks[0])
        np.testing.assert_allclose(actual, expected)
        flattened = flatten_landmarks(actual)
        self.assertEqual(flattened.shape, (63,))
        self.assertEqual(flattened.dtype, np.float32)
        np.testing.assert_allclose(flattened[:6], expected[:2].reshape(-1))

    def test_wrist_origin_and_max_radius(self):
        normalized = normalize_landmarks(make_coordinates(), (640, 480))
        np.testing.assert_array_equal(normalized[0], [0, 0, 0])
        self.assertAlmostEqual(np.linalg.norm(normalized, axis=1).max(), 1.0)

    def test_translation_and_uniform_scale_invariance(self):
        coordinates = make_coordinates()
        baseline = normalize_landmarks(coordinates, (640, 480))
        transformed = coordinates * 2.5 + [0.1, -0.2, 0.3]
        np.testing.assert_allclose(normalize_landmarks(transformed, (640, 480)), baseline,
                                   atol=1e-12)

    def test_aspect_ratio_matches_equal_pixel_displacements(self):
        coordinates = np.zeros((21, 3))
        coordinates[1] = [0.1, 0.2, 0]  # 100 x pixels and 100 y pixels in 1000x500.
        normalized = normalize_landmarks(coordinates, (1000, 500))
        self.assertAlmostEqual(normalized[1, 0], normalized[1, 1])

    def test_input_array_is_not_modified(self):
        coordinates = make_coordinates()
        original = coordinates.copy()
        normalize_landmarks(coordinates, (640, 480))
        np.testing.assert_array_equal(coordinates, original)

    def test_no_hand_returns_126_zeros(self):
        features = create_feature_vector(make_result([]), (640, 480))
        self.assertEqual(features.shape, (126,))
        np.testing.assert_array_equal(features, np.zeros(126))

    def test_single_hand_uses_correct_slot(self):
        for label, occupied, missing in (("Left", slice(0, 63), slice(63, 126)),
                                         ("Right", slice(63, 126), slice(0, 63))):
            with self.subTest(label=label):
                features = create_feature_vector(make_result([label]), (640, 480))
                self.assertEqual(features.shape, (126,))
                self.assertTrue(features[occupied].any())
                self.assertFalse(features[missing].any())

    def test_two_hands_are_independent_of_detection_order(self):
        first = make_result(["Left", "Right"])
        first.hand_landmarks[1][4].z += 0.05
        reversed_result = SimpleNamespace(hand_landmarks=first.hand_landmarks[::-1],
                                          handedness=first.handedness[::-1])
        features = create_feature_vector(first, (640, 480))
        self.assertEqual(features.shape, (126,))
        np.testing.assert_array_equal(features, create_feature_vector(reversed_result, (640, 480)))
        self.assertFalse(np.array_equal(features[:63], features[63:]))

    def test_invalid_arrays_are_rejected(self):
        invalid_arrays = [np.zeros((20, 3)), np.zeros((21, 2)),
                          np.full((21, 3), np.nan), np.full((21, 3), np.inf),
                          np.zeros((21, 3))]
        for coordinates in invalid_arrays:
            with self.subTest(shape=coordinates.shape):
                with self.assertRaises(ValueError):
                    normalize_landmarks(coordinates, (640, 480))

    def test_bad_handedness_and_excess_hands_are_rejected(self):
        for labels in (["Unknown"], ["Left", "Left"], ["Left", "Right", "Left"]):
            with self.subTest(labels=labels):
                with self.assertRaises(ValueError):
                    create_feature_vector(make_result(labels), (640, 480))
        result = make_result(["Left"])
        result.handedness = []
        with self.assertRaises(ValueError):
            create_feature_vector(result, (640, 480))

    def test_incorrect_landmark_count_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_hand_landmarks(make_result(["Left"]).hand_landmarks[0][:-1])

    def test_invalid_frame_size_is_rejected(self):
        for frame_size in ((0, 480), (640, -1), (np.nan, 480)):
            with self.subTest(frame_size=frame_size):
                with self.assertRaises(ValueError):
                    create_feature_vector(make_result([]), frame_size)


if __name__ == "__main__":
    unittest.main()
