"""Read-only dataset inspection, including schema, features and provenance."""

from collections import Counter
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from config import settings
from src.dataset.dataset_utils import CSV_COLUMNS, FEATURE_COLUMNS, current_schema, image_dataset_schema, alphabet_webcam_schema, feature_key


@dataclass
class DatasetReport:
    """Aggregate counts and problems, without keeping the dataset in memory."""

    path: Path
    total_rows: int = 0
    valid_rows: int = 0
    invalid_rows: int = 0
    label_counts: Counter = field(default_factory=Counter)
    participants: set = field(default_factory=set)
    sessions: set = field(default_factory=set)
    sessions_by_label: dict = field(default_factory=dict)
    unverified_rows: int = 0
    duplicate_rows: int = 0
    duplicate_features: int = 0
    conflicting_labels: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    passes_basic_training_checks: bool = False
    vocabulary: list = field(default_factory=lambda: settings.SIGN_LABELS.copy())
    schema: dict = field(default_factory=current_schema)
    source_groups: set = field(default_factory=set)
    groups_by_label: dict = field(default_factory=dict)


def inspect_row(row: dict, vocabulary=None, external_images: bool = False) -> tuple[tuple[float, ...] | None, list[str]]:
    """Check one complete row; return a duplicate key and understandable errors."""
    errors = []
    missing = [name for name, value in row.items() if not value.strip()]
    if missing:
        return None, ["Missing values: " + ", ".join(missing)]
    vocabulary = settings.SIGN_LABELS if vocabulary is None else vocabulary
    if row["label"] not in vocabulary:
        errors.append(f"Unknown label: {row['label']}")
    try:
        if int(row["frame_width"]) <= 0 or int(row["frame_height"]) <= 0:
            raise ValueError
    except ValueError:
        errors.append("Frame width/height must be positive integers.")
    if external_images:
        if any(row[column] != "UNKNOWN" for column in ("participant_id", "session_id", "captured_at_utc")):
            errors.append("External image signer/session/capture time must remain UNKNOWN.")
        if len(row["source_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["source_sha256"]):
            errors.append("Invalid source image SHA-256.")
    else:
        try:
            timestamp = datetime.fromisoformat(row["captured_at_utc"])
            if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
                raise ValueError
        except ValueError:
            errors.append("captured_at_utc must be an ISO timestamp with UTC offset.")
    if row["hand_count"] not in ("1", "2"):
        errors.append("Hand count must be 1 or 2.")
    try:
        values = np.array([float(row[column]) for column in FEATURE_COLUMNS])
        key = feature_key(values)
    except (ValueError, OverflowError) as error:
        errors.append(f"Invalid features: {error}")
        return None, errors

    # Version-1 schema: each occupied slot has wrist 0 and maximum joint radius 1.
    occupied_slots = 0
    for slot_name, slot in zip(("Left", "Right"), values.reshape(2, 21, 3)):
        if not slot.any():
            continue
        occupied_slots += 1
        if not np.allclose(slot[0], 0, atol=1e-5, rtol=0):
            errors.append(f"{slot_name} wrist is not the normalized origin.")
        max_radius = np.linalg.norm(slot, axis=1).max()
        if not np.isclose(max_radius, 1.0, atol=1e-4, rtol=0):
            errors.append(f"{slot_name} hand scale does not match preprocessing version.")
    if str(occupied_slots) != row["hand_count"]:
        errors.append("Hand count disagrees with the populated feature slots.")
    return key, errors


def validate_dataset(path: Path, min_samples: int = settings.MIN_SAMPLES_PER_CLASS,
                     imbalance_ratio: float = settings.CLASS_IMBALANCE_RATIO) -> DatasetReport:
    """Inspect a CSV and sidecar; never change samples or silently drop bad rows."""
    if min_samples < 1 or not 0 < imbalance_ratio <= 1:
        raise ValueError("Minimum samples must be positive; imbalance ratio must be in (0, 1].")
    path = Path(path)
    report = DatasetReport(path)
    metadata_path = path.with_suffix(".metadata.json")
    external_images = False
    columns = CSV_COLUMNS
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("Schema metadata must be a JSON object.")
        candidate_schema = metadata.get("schema", metadata)
        if not isinstance(candidate_schema, dict):
            raise ValueError("Dataset schema must be a JSON object.")
        external_images = candidate_schema.get("dataset_kind") == "external_images"
        alphabet_webcam = candidate_schema.get("dataset_kind") == "alphabet_webcam"
        expected_schema = (image_dataset_schema() if external_images else
                           alphabet_webcam_schema() if alphabet_webcam else current_schema())
        if candidate_schema != expected_schema or settings.FEATURE_COUNT != len(FEATURE_COLUMNS):
            report.errors.append("Schema/preprocessing metadata differs from current settings.")
        report.schema = expected_schema
        if external_images or alphabet_webcam:
            report.vocabulary = settings.ALPHABET_LABELS.copy()
            columns = expected_schema["columns"]
    except (OSError, ValueError) as error:
        report.errors.append(f"Missing/unreadable schema metadata: {error}")

    seen_rows = set()
    seen_features = {}
    session_participants = {}
    group_labels = {}
    seen_source_hashes = set()
    detail_count = 0
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, None)
        if header != columns:
            feature_count = sum(name.startswith("f") and name[1:].isdigit() for name in header or [])
            report.errors.append(f"Incorrect CSV header: expected {len(FEATURE_COLUMNS)} features "
                                 f"and {len(columns)} total columns; found {feature_count} feature columns.")
            return report
        try:
            for record_number, values in enumerate(reader, start=2):
                report.total_rows += 1
                row_errors = []
                if len(values) != len(columns):
                    row_errors.append(f"Incorrect feature/row length: expected {len(columns)} "
                                      f"columns, found {len(values)}.")
                else:
                    row = dict(zip(columns, values))
                    key, row_errors = inspect_row(row, report.vocabulary, external_images)
                    if external_images:
                        group = row["source_group_id"]
                        previous_label = group_labels.setdefault(group, row["label"])
                        if previous_label != row["label"]:
                            row_errors.append("Source group spans conflicting labels.")
                        if row["source_sha256"] in seen_source_hashes:
                            row_errors.append("Repeated source image hash.")
                        seen_source_hashes.add(row["source_sha256"])
                    row_key = tuple(values)
                    if row_key in seen_rows:
                        report.duplicate_rows += 1
                        row_errors.append("Exact duplicate CSV row.")
                    seen_rows.add(row_key)
                    if key is not None:
                        if key in seen_features:
                            report.duplicate_features += 1
                            previous_label = seen_features[key]
                            if previous_label == row["label"]:
                                row_errors.append("Duplicate rounded features for the same label.")
                            else:
                                report.conflicting_labels += 1
                                row_errors.append("Identical features have conflicting labels.")
                        else:
                            seen_features[key] = row["label"]
                    session = row["session_id"]
                    participant = row["participant_id"]
                    if session and participant:
                        previous_participant = session_participants.setdefault(session, participant)
                        if previous_participant != participant:
                            row_errors.append("Session ID is shared by different participants.")
                    if not row_errors:
                        label = row["label"]
                        report.label_counts[label] += 1
                        if external_images:
                            report.source_groups.add(row["source_group_id"])
                            report.groups_by_label.setdefault(label, set()).add(row["source_group_id"])
                        else:
                            report.participants.add(participant)
                            report.sessions.add(session)
                            report.sessions_by_label.setdefault(label, set()).add(session)
                        if external_images or row["source_reference"].strip().upper() == "UNVERIFIED":
                            report.unverified_rows += 1
                if row_errors:
                    report.invalid_rows += 1
                    if detail_count < 20:
                        report.errors.append(f"Record {record_number}: " + " ".join(row_errors))
                        detail_count += 1
                else:
                    report.valid_rows += 1
        except csv.Error as error:
            report.errors.append(f"CSV parsing failed near physical line {reader.line_num}: {error}")
    if report.invalid_rows > detail_count:
        report.errors.append(f"{report.invalid_rows - detail_count} additional invalid rows; first 20 shown.")
    if not report.valid_rows:
        report.errors.append("No valid unique samples found. Collect examples before training.")

    present_labels = list(report.label_counts)
    missing_labels = [label for label in report.vocabulary if label not in present_labels]
    if missing_labels:
        report.warnings.append("Configured labels with no valid samples: " + ", ".join(missing_labels))
    small_labels = [f"{label} ({count})" for label, count in report.label_counts.items() if count < min_samples]
    if small_labels:
        report.warnings.append(f"Below advisory minimum of {min_samples} samples: " + ", ".join(small_labels))
    if len(present_labels) < 2:
        report.warnings.append("At least two populated classes are needed for sign classification.")
    elif min(report.label_counts.values()) / max(report.label_counts.values()) < imbalance_ratio:
        report.warnings.append(f"Class imbalance: smallest/largest count is below {imbalance_ratio:.2f}.")
    if report.unverified_rows:
        report.warnings.append(f"{report.unverified_rows} valid samples are marked UNVERIFIED; keep practice data separate.")
    if external_images:
        report.warnings.append("External alphabet labels are not independently verified; signer/session identities "
                               "and licence documentation are unknown. Source groups are not sessions.")
    elif report.valid_rows and len(report.participants) < 2:
        report.warnings.append("Only one participant: this cannot establish generalization to other signers.")
    single_sessions = [label for label, sessions in report.sessions_by_label.items() if len(sessions) < 2]
    if single_sessions:
        report.warnings.append("Only one session for these labels: " + ", ".join(single_sessions)
                               + ". Collect separate sessions for meaningful held-out evaluation.")
    report.passes_basic_training_checks = (
        not report.errors and len(present_labels) >= 2 and not report.unverified_rows
        and all(count >= min_samples for count in report.label_counts.values())
        and all(len(sessions) >= 2 for sessions in report.sessions_by_label.values())
    )
    return report


def format_report(report: DatasetReport) -> str:
    """Produce the same plain-language report for the terminal and optional file."""
    lines = ["Dataset Summary", f"File: {report.path}", "", "Label          Valid unique samples"]
    for label in report.vocabulary:
        lines.append(f"{label:<14} {report.label_counts.get(label, 0)}")
    lines += ["", f"Total CSV rows: {report.total_rows}", f"Valid unique samples: {report.valid_rows}",
              f"Invalid rows: {report.invalid_rows}", f"Exact duplicate rows: {report.duplicate_rows}",
              f"Duplicate feature vectors: {report.duplicate_features}",
              f"Conflicting-label duplicates: {report.conflicting_labels}",
              ("Participants: UNKNOWN | Sessions: UNKNOWN" if report.schema.get("dataset_kind") == "external_images"
               else f"Participants: {len(report.participants)} | Sessions: {len(report.sessions)}"),
              f"UNVERIFIED valid samples: {report.unverified_rows}", "",
              "Integrity: " + ("PASS" if not report.errors else "FAIL"),
              "Basic training checks: " + ("PASS" if report.passes_basic_training_checks else "NOT MET")]
    if report.schema.get("dataset_kind") == "external_images":
        lines.append(f"Filename/content source groups: {len(report.source_groups)} (signer/session counts unknown)")
    for error in report.errors:
        lines.append("ERROR: " + error)
    for warning in report.warnings:
        lines.append("WARNING: " + warning)
    lines.append("Reference notes and sample counts do not prove FSL correctness or model accuracy.")
    return "\n".join(lines)
