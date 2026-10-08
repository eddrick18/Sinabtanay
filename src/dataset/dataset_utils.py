"""CSV persistence and compatibility checks needed to safely append samples."""

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from config import settings
from src.hand_tracking.landmark_processor import FEATURE_COUNT

CONTEXT_COLUMNS = ["label", "participant_id", "session_id", "captured_at_utc",
                   "hand_count", "frame_width", "frame_height", "source_reference"]
FEATURE_COLUMNS = [f"f{index}" for index in range(1, FEATURE_COUNT + 1)]
CSV_COLUMNS = CONTEXT_COLUMNS + FEATURE_COLUMNS
IMAGE_CONTEXT_COLUMNS = CONTEXT_COLUMNS + ["source_file", "source_group_id", "source_sha256"]
IMAGE_CSV_COLUMNS = IMAGE_CONTEXT_COLUMNS + FEATURE_COLUMNS


def feature_key(features: np.ndarray) -> tuple[float, ...]:
    """Canonical six-decimal values for storage and exact duplicate comparison."""
    values = np.asarray(features, dtype=np.float64)
    if values.shape != (FEATURE_COUNT,) or not np.isfinite(values).all():
        raise ValueError(f"Expected {FEATURE_COUNT} finite feature values.")
    if not values.any():
        raise ValueError("All-zero features cannot be saved as a hand sample.")
    rounded = tuple(float(f"{value:.6f}") for value in values)
    if not any(rounded):
        raise ValueError("Features round to an empty hand sample.")
    return rounded


def current_schema() -> dict:
    """Describe the feature contract; a changed contract requires a new dataset."""
    return {
        "schema_version": 1,
        "preprocessing_version": settings.PREPROCESSING_VERSION,
        "feature_count": FEATURE_COUNT,
        "hand_order": ["Left", "Right"],
        "mirror_camera": settings.MIRROR_CAMERA,
        "columns": CSV_COLUMNS,
        "feature_decimal_places": 6,
    }


def alphabet_webcam_schema() -> dict:
    """Real recording context, with a separate static alphabet vocabulary."""
    schema = current_schema()
    schema.update(schema_version=3, dataset_kind="alphabet_webcam", vocabulary=settings.ALPHABET_LABELS)
    return schema


def image_dataset_schema() -> dict:
    """Same numeric contract, separate provenance columns and alphabet vocabulary."""
    schema = current_schema()
    schema.update(schema_version=2, dataset_kind="external_images", columns=IMAGE_CSV_COLUMNS,
                  vocabulary=settings.ALPHABET_LABELS,
                  group_semantics="filename-and-content-components; not recording sessions")
    return schema


class DatasetWriter:
    """One collector at a time: append validated rows and remember duplicate keys."""

    def __init__(self, path: Path, vocabulary: str = "words") -> None:
        if vocabulary not in ("words", "alphabet"):
            raise ValueError("Vocabulary must be words or alphabet.")
        self.labels = settings.ALPHABET_LABELS if vocabulary == "alphabet" else settings.SIGN_LABELS
        self.path = Path(path)
        self.metadata_path = self.path.with_suffix(".metadata.json")
        self.sample_counts = {}
        self.session_counts = {}
        self.seen_features = {}
        if settings.FEATURE_COUNT != FEATURE_COUNT:
            raise ValueError("Configured feature count does not match the processor.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        expected_schema = alphabet_webcam_schema() if vocabulary == "alphabet" else current_schema()
        if self.metadata_path.exists():
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if metadata != expected_schema:
                raise ValueError("Dataset schema/preprocessing mismatch. Use a new dataset path.")
        elif self.path.exists() and self.path.stat().st_size:
            raise ValueError("Existing dataset has no schema metadata; refusing to guess its format.")
        else:
            temporary_path = self.metadata_path.with_suffix(".json.tmp")
            temporary_path.write_text(json.dumps(expected_schema, indent=2) + "\n", encoding="utf-8")
            temporary_path.replace(self.metadata_path)

        if self.path.exists() and self.path.stat().st_size:
            self._read_existing_samples()
        else:
            with self.path.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(CSV_COLUMNS)

    def _read_existing_samples(self) -> None:
        """Check existing rows before appending; full dataset reports come in Phase 5."""
        with self.path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != CSV_COLUMNS:
                raise ValueError("Dataset CSV header differs from the expected schema.")
            for line_number, row in enumerate(reader, start=2):
                try:
                    if None in row or any(value is None or value == "" for value in row.values()):
                        raise ValueError("Incorrect column count or empty value.")
                    if row["label"] not in self.labels:
                        raise ValueError("Label is not in configured vocabulary.")
                    key = feature_key(np.array([float(row[column]) for column in FEATURE_COLUMNS]))
                    if row["hand_count"] not in ("1", "2"):
                        raise ValueError("Hand count must be 1 or 2.")
                    if int(row["frame_width"]) <= 0 or int(row["frame_height"]) <= 0:
                        raise ValueError("Invalid image size.")
                    if key in self.seen_features and self.seen_features[key] != row["label"]:
                        raise ValueError("Identical features have contradictory labels.")
                    self.seen_features[key] = row["label"]
                    self._increment_counts(row["label"], row["session_id"])
                except (ValueError, TypeError) as error:
                    raise ValueError(f"Invalid dataset row {line_number}: {error}") from error

    def _increment_counts(self, label: str, session_id: str) -> None:
        self.sample_counts[label] = self.sample_counts.get(label, 0) + 1
        session_key = (label, session_id)
        self.session_counts[session_key] = self.session_counts.get(session_key, 0) + 1

    def save(self, features: np.ndarray, *, label: str, participant_id: str,
             session_id: str, hand_count: int, frame_size: tuple[int, int],
             source_reference: str) -> str:
        """Save one example; return a status, leaving rejected examples out of CSV."""
        if label not in self.labels:
            raise ValueError("Unknown label. Edit SIGN_LABELS after verifying your vocabulary.")
        if hand_count not in (1, 2):
            raise ValueError("Only one/two-hand examples can be saved.")
        if not participant_id.strip() or not session_id.strip() or not source_reference.strip():
            raise ValueError("Participant, session, and reference fields must be nonempty.")
        width, height = frame_size
        if width <= 0 or height <= 0:
            raise ValueError("Invalid frame dimensions.")
        key = feature_key(features)
        if key in self.seen_features:
            if self.seen_features[key] == label:
                return "Skipped: duplicate sample"
            return "Skipped: identical features already have a different label"

        captured_at = datetime.now(timezone.utc).isoformat()
        row = [label, participant_id, session_id, captured_at, hand_count, width, height,
               source_reference] + [f"{value:.6f}" for value in key]
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        self.seen_features[key] = label
        self._increment_counts(label, session_id)
        return "Saved sample"
