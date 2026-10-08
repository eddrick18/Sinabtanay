"""Adapt OpenCV frames to MediaPipe and draw its hand-location results."""

import os

from config import settings

# MediaPipe imports matplotlib; keep its cache inside this project, not the home folder.
os.environ.setdefault("MPLCONFIGDIR", str(settings.PROJECT_ROOT / ".cache" / "matplotlib"))

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

# Landmark indices follow MediaPipe's 21-point hand skeleton.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


class HandDetector:
    """Own one MediaPipe detector for independent images or ordered video frames."""

    def __init__(self, running_mode=vision.RunningMode.VIDEO) -> None:
        if not settings.HAND_MODEL_PATH.is_file():
            raise FileNotFoundError(
                "Hand landmark model missing. Run: python scripts/download_hand_model.py"
            )
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(settings.HAND_MODEL_PATH)),
            running_mode=running_mode,
            num_hands=settings.MAX_HANDS,
            min_hand_detection_confidence=settings.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=settings.MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=settings.MIN_TRACKING_CONFIDENCE,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.running_mode = running_mode
        self.last_timestamp_ms = -1

    def detect(self, frame: np.ndarray, timestamp_ms: int | None = None) -> vision.HandLandmarkerResult:
        """Receive BGR pixels and monotonic milliseconds; return hand locations."""
        # MediaPipe requires strictly increasing timestamps, even for fast frames.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        if self.running_mode == vision.RunningMode.IMAGE:
            return self.landmarker.detect(image)
        if timestamp_ms is None:
            raise ValueError("Video detection requires a monotonic timestamp.")
        timestamp_ms = max(timestamp_ms, self.last_timestamp_ms + 1)
        result = self.landmarker.detect_for_video(image, timestamp_ms)
        self.last_timestamp_ms = timestamp_ms
        return result

    def close(self) -> None:
        """Release the MediaPipe task's native resources."""
        self.landmarker.close()


def get_hand_labels(result: vision.HandLandmarkerResult) -> list[str]:
    """Return one handedness label per detected hand, with a safe fallback."""
    labels = []
    for index in range(len(result.hand_landmarks)):
        categories = result.handedness[index] if index < len(result.handedness) else []
        label = categories[0].category_name if categories else "Unknown"
        labels.append(label or "Unknown")
    return labels


def draw_hand_results(frame: np.ndarray, result: vision.HandLandmarkerResult) -> None:
    """Draw landmarks/connections and handedness in place on a BGR image."""
    height, width = frame.shape[:2]
    labels = get_hand_labels(result)
    for landmarks, label in zip(result.hand_landmarks, labels):
        # Image-normalized x/y become pixel locations; this is drawing, not ML preprocessing.
        points = [(int(point.x * width), int(point.y * height)) for point in landmarks]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0, 220, 0), 2)
        for point in points:
            cv2.circle(frame, point, 4, (0, 0, 255), -1)
        wrist_x, wrist_y = points[0]
        cv2.putText(frame, label, (max(0, wrist_x), max(20, wrist_y - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
