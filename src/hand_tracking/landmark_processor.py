"""Pure numerical preprocessing shared by future collection and inference."""

from collections.abc import Sequence
from typing import Any

import numpy as np

LANDMARK_COUNT = 21
COORDINATE_COUNT = 3
VALUES_PER_HAND = LANDMARK_COUNT * COORDINATE_COUNT
FEATURE_COUNT = 2 * VALUES_PER_HAND
HAND_ORDER = ("Left", "Right")


def extract_hand_landmarks(landmarks: Sequence[Any]) -> np.ndarray:
    """Convert 21 MediaPipe-like x/y/z objects into a finite (21, 3) array."""
    if len(landmarks) != LANDMARK_COUNT:
        raise ValueError("Each hand must contain exactly 21 landmarks.")
    try:
        coordinates = np.array([[point.x, point.y, point.z] for point in landmarks],
                               dtype=np.float64)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("Each landmark must have numerical x, y, z coordinates.") from error
    return validate_coordinates(coordinates)


def validate_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Validate shape and finite values without modifying the caller's array."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.shape != (LANDMARK_COUNT, COORDINATE_COUNT):
        raise ValueError("Landmark array must have shape (21, 3).")
    if not np.isfinite(coordinates).all():
        raise ValueError("Landmark coordinates must be finite (no NaN or infinity).")
    return coordinates


def normalize_landmarks(coordinates: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
    """Subtract wrist and divide by maximum 3D radius in image-width units.

    frame_size is (width, height). Image x and MediaPipe z use width units;
    multiplying y by height/width corrects for rectangular image dimensions.
    This does not recover metric 3D geometry or normalize rotation.
    """
    coordinates = validate_coordinates(coordinates)
    width, height = frame_size
    if not np.isfinite([width, height]).all() or width <= 0 or height <= 0:
        raise ValueError("Frame width and height must be finite and positive.")
    relative_coordinates = coordinates - coordinates[0]
    relative_coordinates[:, 1] *= height / width
    scale = float(np.max(np.linalg.norm(relative_coordinates, axis=1)))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("Hand scale is zero, too small, or invalid; frame is unusable.")
    return relative_coordinates / scale


def flatten_landmarks(normalized_landmarks: np.ndarray) -> np.ndarray:
    """Return 63 float32 values ordered x0,y0,z0,x1,y1,z1,...,x20,y20,z20."""
    return validate_coordinates(normalized_landmarks).reshape(VALUES_PER_HAND).astype(np.float32)


def create_feature_vector(result: Any, frame_size: tuple[int, int]) -> np.ndarray:
    """Return Left then Right slots; pad absent hands and reject ambiguous input.

    Empty results produce zeros for display/testing, never a valid training sample.
    The caller must separately check that at least one hand was detected.
    """
    width, height = frame_size
    if not np.isfinite([width, height]).all() or width <= 0 or height <= 0:
        raise ValueError("Frame width and height must be finite and positive.")
    hands = result.hand_landmarks
    handedness = result.handedness
    if len(hands) > 2:
        raise ValueError("Only up to two hands are supported by this feature schema.")
    if len(hands) != len(handedness):
        raise ValueError("Each detected hand requires handedness for consistent slot ordering.")

    features = np.zeros(FEATURE_COUNT, dtype=np.float32)
    occupied_slots = set()
    for landmarks, categories in zip(hands, handedness):
        label = categories[0].category_name if categories else None
        if label not in HAND_ORDER:
            raise ValueError("Handedness must be Left or Right; do not guess a feature slot.")
        if label in occupied_slots:
            raise ValueError(f"Duplicate {label} handedness; frame has ambiguous hand slots.")
        occupied_slots.add(label)
        coordinates = extract_hand_landmarks(landmarks)
        normalized = normalize_landmarks(coordinates, frame_size)
        hand_features = flatten_landmarks(normalized)
        start = HAND_ORDER.index(label) * VALUES_PER_HAND
        features[start:start + VALUES_PER_HAND] = hand_features
    return features
