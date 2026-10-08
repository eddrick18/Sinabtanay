"""Import independent alphabet images with explicit unknown provenance and audit."""

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import tempfile

import cv2
import numpy as np

from config import settings
from src.dataset.dataset_utils import IMAGE_CSV_COLUMNS, feature_key, image_dataset_schema
from src.hand_tracking.hand_detector import HandDetector, mp, vision
from src.hand_tracking.landmark_processor import create_feature_vector

logger = logging.getLogger(__name__)
AUDIT_COLUMNS = ["source_file", "label", "source_sha256", "source_group_id", "status", "detail"]


def source_name_key(path: Path) -> tuple[str, str]:
    """Conservative filename family: original, *_jpg.rf.*, and '- Copy' variants."""
    base = path.name.split("_jpg.rf.")[0] if "_jpg.rf." in path.name else path.stem
    base = re.sub(r" - Copy(?: \(\d+\))?$", "", base, flags=re.IGNORECASE)
    return str(path.parent), base


def build_source_groups(entries: list[dict]) -> None:
    """Connect filename families and identical bytes; do not infer signer/session IDs.

    A small union-find tracks transitive connections (A shares a name with B,
    B shares bytes with C => all three stay in one evaluation group).
    """
    parents = list(range(len(entries)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(first, second):
        parents[find(second)] = find(first)

    names, hashes = {}, {}
    for index, entry in enumerate(entries):
        key = source_name_key(entry["path"])
        if key in names:
            join(index, names[key])
        names[key] = index
        digest = entry["source_sha256"]
        if digest in hashes:
            join(index, hashes[digest])
        hashes[digest] = index
    components = defaultdict(list)
    for index in range(len(entries)):
        components[find(index)].append(index)
    for members in components.values():
        canonical_path = min(entries[index]["source_file"] for index in members)
        group = "source_" + hashlib.sha256(canonical_path.encode("utf-8")).hexdigest()[:24]
        labels = {entries[index]["label"] for index in members}
        for index in members:
            entries[index]["source_group_id"] = group
            if len(labels) > 1:
                entries[index]["status"] = "conflicting_source_labels"
                entries[index]["detail"] = "Connected source family spans labels: " + ", ".join(sorted(labels))


def output_paths(dataset_path: Path) -> dict[str, Path]:
    return {
        "dataset": dataset_path,
        "metadata": dataset_path.with_suffix(".metadata.json"),
        "audit": dataset_path.with_suffix(".audit.csv"),
        "report": dataset_path.with_suffix(".import_report.json"),
    }


def merge_identical_feature_families(entries: list[dict], candidates: list[dict]) -> None:
    """Conservatively link source families whose extracted vectors are identical."""
    links = {entry["source_group_id"]: entry["source_group_id"] for entry in entries}

    def root(group):
        while links[group] != group:
            links[group] = links[links[group]]
            group = links[group]
        return group

    seen = {}
    for candidate in candidates:
        group = candidate["entry"]["source_group_id"]
        key = candidate["key"]
        if key in seen:
            first, second = sorted((root(group), root(seen[key])))
            links[second] = first
        seen[key] = group
    group_labels = defaultdict(set)
    for entry in entries:
        entry["source_group_id"] = root(entry["source_group_id"])
        group_labels[entry["source_group_id"]].add(entry["label"])
    for candidate in candidates:
        entry = candidate["entry"]
        if len(group_labels[entry["source_group_id"]]) > 1:
            entry.update(status="conflicting_feature_labels", detail="Feature-linked source family spans labels")


def import_images(source_dir: Path, dataset_path: Path) -> dict:
    """Run image-mode extraction, withhold conflicts, stage outputs and publish."""
    source_dir, dataset_path = Path(source_dir).resolve(), Path(dataset_path).resolve()
    paths = output_paths(dataset_path)
    if any(path.exists() for path in paths.values()):
        raise ValueError("Import outputs already exist. Use a new --output path to preserve this dataset.")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source image folder not found: {source_dir}")
    files = sorted(path for path in source_dir.rglob("*")
                   if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        raise ValueError("No supported images found under the source folder.")
    entries = []
    for index, path in enumerate(files):
        relative_path = path.relative_to(source_dir)
        label = relative_path.parts[0] if len(relative_path.parts) > 1 else "UNKNOWN"
        entries.append({"path": path, "source_file": relative_path.as_posix(), "label": label,
                        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "source_group_id": "", "status": "pending", "detail": ""})
        if (index + 1) % 500 == 0:
            logger.info("Hashed %s/%s source files.", index + 1, len(files))
    build_source_groups(entries)
    logger.info("Source grouping complete; starting independent IMAGE-mode extraction.")
    byte_representatives = {}
    candidates = []
    detector = HandDetector(running_mode=vision.RunningMode.IMAGE)
    try:
        for index, entry in enumerate(entries):
            label = entry["label"]
            if label in ("J", "Z"):
                entry.update(status="excluded_dynamic_letter", detail="Not supported by static baseline")
            elif label not in settings.ALPHABET_LABELS:
                entry.update(status="unsupported_label", detail="Not in alphabet vocabulary")
            elif entry["status"] == "conflicting_source_labels":
                pass
            elif entry["source_sha256"] in byte_representatives:
                entry.update(status="duplicate_image", detail=byte_representatives[entry["source_sha256"]])
            else:
                raw_bytes = entry["path"].read_bytes()
                if hashlib.sha256(raw_bytes).hexdigest() != entry["source_sha256"]:
                    raise ValueError("Source image changed during import: " + entry["source_file"])
                try:
                    frame = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR) if raw_bytes else None
                except cv2.error:
                    frame = None
                if frame is None or frame.size == 0:
                    entry.update(status="unreadable_image", detail="OpenCV could not decode pixels")
                else:
                    # Remember even failures: identical bytes produce the same deterministic input.
                    byte_representatives[entry["source_sha256"]] = entry["source_file"]
                    if settings.MIRROR_CAMERA:
                        frame = cv2.flip(frame, 1)
                    result = detector.detect(frame)
                    if not result.hand_landmarks:
                        entry.update(status="no_hand", detail="No hand at configured thresholds")
                    else:
                        height, width = frame.shape[:2]
                        try:
                            features = create_feature_vector(result, (width, height))
                            key = feature_key(features)
                        except ValueError as error:
                            entry.update(status="invalid_landmarks", detail=str(error))
                        else:
                            entry.update(status="candidate")
                            candidates.append({"entry": entry, "key": key, "hand_count": len(result.hand_landmarks),
                                               "width": width, "height": height})
            if (index + 1) % 250 == 0:
                counts = Counter(item["status"] for item in entries[:index + 1])
                logger.info("Processed %s/%s images | candidates %s | no hand %s.",
                            index + 1, len(entries), counts["candidate"], counts["no_hand"])
    finally:
        detector.close()

    # Review feature duplicates globally before publishing anything, so BOTH labels
    # are withheld if the same vector has conflicting labels later in the import.
    merge_identical_feature_families(entries, candidates)
    feature_groups = defaultdict(list)
    for candidate in candidates:
        if candidate["entry"]["status"] == "candidate":
            feature_groups[candidate["key"]].append(candidate)
    retained = []
    for group in feature_groups.values():
        labels = {candidate["entry"]["label"] for candidate in group}
        if len(labels) > 1:
            for candidate in group:
                candidate["entry"].update(status="conflicting_feature_labels", detail="Same features under multiple labels")
        else:
            retained.append(group[0])
            group[0]["entry"].update(status="saved")
            for candidate in group[1:]:
                candidate["entry"].update(status="duplicate_features", detail=group[0]["entry"]["source_file"])
    retained.sort(key=lambda candidate: candidate["entry"]["source_file"])
    report = {
        "source_files": len(entries), "retained_samples": len(retained),
        "status_counts": dict(Counter(entry["status"] for entry in entries)),
        "per_class": {label: dict(Counter(entry["status"] for entry in entries if entry["label"] == label))
                      for label in sorted({entry["label"] for entry in entries})},
        "retained_class_counts": dict(Counter(candidate["entry"]["label"] for candidate in retained)),
        "retained_source_groups": len({candidate["entry"]["source_group_id"] for candidate in retained}),
        "scope": "Dataset-supplied alphabet labels; not independently verified FSL",
    }
    inventory = "\n".join(entry["source_file"] + ":" + entry["source_sha256"] for entry in entries)
    metadata = {
        "schema": image_dataset_schema(),
        "import": {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "source_root": str(source_dir), "source_reference": settings.ALPHABET_SOURCE_URL,
                   "source_inventory_sha256": hashlib.sha256(inventory.encode()).hexdigest(),
                   "signer_ids": "UNKNOWN", "session_ids": "UNKNOWN", "capture_times": "UNKNOWN",
                   "licence": "UNKNOWN - not bundled; check Kaggle metadata",
                   "label_verification": "DATASET_SUPPLIED_NOT_INDEPENDENTLY_VERIFIED",
                   "mediapipe_version": mp.__version__, "running_mode": "IMAGE",
                   "model_sha256": hashlib.sha256(settings.HAND_MODEL_PATH.read_bytes()).hexdigest(),
                   "detection_confidence": settings.MIN_DETECTION_CONFIDENCE,
                   "presence_confidence": settings.MIN_HAND_PRESENCE_CONFIDENCE,
                   "excluded_letters": ["J", "Z"]},
        "summary": report,
    }
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="image-import-", dir=dataset_path.parent) as temporary_directory:
        staged = Path(temporary_directory)
        with (staged / paths["dataset"].name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(IMAGE_CSV_COLUMNS)
            for candidate in retained:
                entry = candidate["entry"]
                writer.writerow([entry["label"], "UNKNOWN", "UNKNOWN", "UNKNOWN", candidate["hand_count"],
                                 candidate["width"], candidate["height"], settings.ALPHABET_SOURCE_URL,
                                 entry["source_file"], entry["source_group_id"], entry["source_sha256"]]
                                + [f"{value:.6f}" for value in candidate["key"]])
        with (staged / paths["audit"].name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=AUDIT_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(entries)
        (staged / paths["report"].name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (staged / paths["metadata"].name).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        for name in ("dataset", "audit", "report", "metadata"):
            # Create in the destination so Windows inherits project permissions.
            with (staged / paths[name].name).open("rb") as source, paths[name].open("xb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
    logger.info("Import finished: %s/%s samples saved; %s source groups.",
                len(retained), len(entries), report["retained_source_groups"])
    return report
