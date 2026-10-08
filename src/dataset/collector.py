"""Collection policy independent of webcam/UI so rejection logic is testable."""

from typing import Any

from config import settings
from src.dataset.dataset_utils import DatasetWriter
from src.hand_tracking.landmark_processor import create_feature_vector


class SampleCollector:
    """Turn explicit capture requests into valid CSV examples."""

    def __init__(self, writer: DatasetWriter, label: str, participant_id: str,
                 session_id: str, source_reference: str, target: int) -> None:
        if label not in writer.labels or target <= 0:
            raise ValueError("Use a configured label and a positive sample target.")
        self.writer = writer
        self.label = label
        self.participant_id = participant_id
        self.session_id = session_id
        self.source_reference = source_reference
        self.target = target
        self.last_capture_time = float("-inf")

    @property
    def session_count(self) -> int:
        return self.writer.session_counts.get((self.label, self.session_id), 0)

    def capture(self, result: Any, frame_size: tuple[int, int], now: float) -> str:
        """Called on Space only; now is monotonic seconds supplied by the UI."""
        if now - self.last_capture_time < settings.CAPTURE_COOLDOWN_SECONDS:
            return "Wait: capture cooldown"
        self.last_capture_time = now
        if self.session_count >= self.target:
            return "Session target reached; Q to quit"
        if not result.hand_landmarks:
            return "Skipped: no hand detected"
        try:
            features = create_feature_vector(result, frame_size)
        except ValueError as error:
            return f"Skipped: {error}"
        return self.writer.save(features, label=self.label, participant_id=self.participant_id,
                                session_id=self.session_id, hand_count=len(result.hand_landmarks),
                                frame_size=frame_size, source_reference=self.source_reference)
