"""Validate numerical examples, split them, fit a forest, and save one run."""

from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import math
from pathlib import Path
import platform
import shutil
import tempfile

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from config import settings
from src.dataset.dataset_utils import FEATURE_COLUMNS, current_schema
from src.dataset.validator import validate_dataset
from src.training.evaluator import evaluate_predictions, save_confusion_matrix

logger = logging.getLogger(__name__)


def load_training_data(path: Path, practice: bool = False, minimum_classes: int = 2) -> tuple[pd.DataFrame, dict]:
    """Validate before reading; keep context as strings and X limited to feature columns."""
    snapshot = path.read_bytes()
    report = validate_dataset(path)
    if report.errors:
        raise ValueError("Dataset integrity failed. Run scripts/check_dataset.py. " + report.errors[0])
    if len(report.label_counts) < minimum_classes:
        raise ValueError("Training needs at least two populated labels. The one-class practice file "
                         "only tests capture; collect more classes in a separate dataset.")
    if report.unverified_rows and not practice:
        raise ValueError("UNVERIFIED data is not real FSL training data. Verify actual signs or use "
                         "--practice for an explicitly separate software experiment.")
    if path.read_bytes() != snapshot:
        raise ValueError("Dataset changed during validation. Close the collector and retry.")
    for warning in report.warnings:
        logger.warning(warning)
    frame = pd.read_csv(io.BytesIO(snapshot), dtype=str, keep_default_na=False)
    provenance = {
        "dataset_sha256": hashlib.sha256(snapshot).hexdigest(),
        "dataset_warnings": report.warnings,
        "unverified_samples": report.unverified_rows,
        "dataset_samples": report.valid_rows,
        "schema": report.schema,
    }
    return frame, provenance


def split_dataset(frame: pd.DataFrame, mode: str, test_size: float,
                  random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """Split row indices; session mode stratifies one-label recording sessions."""
    if not 0 < test_size < 1:
        raise ValueError("Test size must be between 0 and 1.")
    labels = frame["label"].to_numpy()
    class_count = len(set(labels))
    if class_count < 2:
        raise ValueError("At least two classes are required.")
    if mode == "frame":
        try:
            train_indices, test_indices = train_test_split(np.arange(len(frame)), test_size=test_size,
                                                           random_state=random_state, stratify=labels)
        except ValueError as error:
            raise ValueError("Stratified frame split is too small. Collect more examples per class "
                             "or adjust --test-size. " + str(error)) from error
    elif mode in ("session", "group"):
        group_column = "session_id" if mode == "session" else "source_group_id"
        if group_column not in frame:
            raise ValueError("Source-group split requires an imported image dataset.")
        if mode == "session" and (frame["session_id"] == "UNKNOWN").any():
            raise ValueError("Unknown recording sessions cannot support session holdout. Use --split group for imported images.")
        session_label_counts = frame.groupby(group_column)["label"].nunique()
        if (session_label_counts != 1).any():
            raise ValueError("Session/group split expects one label per session/group, as produced by our collector. "
                             "Mixed-label sessions require a different grouped split strategy.")
        sessions = frame[[group_column, "label"]].drop_duplicates().sort_values(group_column)
        if sessions["label"].value_counts().min() < 2:
            raise ValueError("Session/group split needs at least two separate sessions/groups for EVERY present label. "
                             "Collect additional independent data.")
        test_session_count = max(class_count, math.ceil(len(sessions) * test_size))
        if len(sessions) - test_session_count < class_count:
            raise ValueError("Requested test fraction leaves too few training sessions per class.")
        train_sessions, test_sessions = train_test_split(sessions[group_column].to_numpy(),
                                                        test_size=test_session_count,
                                                        random_state=random_state,
                                                        stratify=sessions["label"].to_numpy())
        train_indices = np.flatnonzero(frame[group_column].isin(train_sessions).to_numpy())
        test_indices = np.flatnonzero(frame[group_column].isin(test_sessions).to_numpy())
    else:
        raise ValueError("Split mode must be session, group or frame.")
    if set(train_indices) & set(test_indices):
        raise ValueError("Train/test rows overlap.")
    if set(labels[train_indices]) != set(labels) or set(labels[test_indices]) != set(labels):
        raise ValueError("Every class must appear in both splits. Collect more samples/sessions.")
    return train_indices, test_indices


def save_training_run(model, metadata: dict, evaluation: dict, output_dir: Path) -> None:
    """Stage outputs, refuse overwrites, and publish metadata last as the run record."""
    relative_paths = [Path("trained/sign_classifier.joblib"), Path("metadata/confusion_matrix.png"),
                      Path("metadata/classification_report.txt"), Path("metadata/evaluation.json"),
                      Path("metadata/model_metadata.json")]
    if any((output_dir / path).exists() for path in relative_paths):
        raise ValueError("Training outputs already exist. Use a new --output-dir to preserve this run.")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="training-", dir=output_dir) as temporary_directory:
        staged = Path(temporary_directory)
        joblib.dump(model, staged / "sign_classifier.joblib")
        metadata["model_sha256"] = hashlib.sha256((staged / "sign_classifier.joblib").read_bytes()).hexdigest()
        save_confusion_matrix(evaluation, staged / "confusion_matrix.png", metadata["practice"])
        (staged / "classification_report.txt").write_text(
            ("PRACTICE ONLY - not FSL accuracy\n" if metadata["practice"] else "")
            + evaluation["classification_report_text"], encoding="utf-8")
        (staged / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
        (staged / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        for relative_path in relative_paths:
            destination = output_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            # New destination files inherit the user's project permissions on Windows.
            # Renaming tempfile artifacts preserves their restrictive sandbox ACLs.
            with (staged / relative_path.name).open("rb") as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)


def train_model(dataset_path: Path, output_dir: Path, *, split_mode: str = "auto",
                test_size: float = settings.TEST_SIZE, random_state: int = settings.RANDOM_STATE,
                tree_count: int = settings.RANDOM_FOREST_TREES, practice: bool = False) -> dict:
    """Fit using training rows only; evaluate once on held-out rows, then persist."""
    dataset_path, output_dir = Path(dataset_path), Path(output_dir)
    if tree_count < 1:
        raise ValueError("Tree count must be positive.")
    if practice and output_dir.resolve() == (settings.PROJECT_ROOT / "models").resolve():
        raise ValueError("Practice output must be separate from the default real model directory.")
    frame, provenance = load_training_data(dataset_path, practice)
    external_images = provenance["schema"].get("dataset_kind") == "external_images"
    if split_mode == "auto":
        split_mode = "group" if external_images else "session"
    train_indices, test_indices = split_dataset(frame, split_mode, test_size, random_state)
    features = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    labels = frame["label"].to_numpy()
    train_sessions = set(frame.iloc[train_indices]["session_id"])
    test_sessions = set(frame.iloc[test_indices]["session_id"])
    train_participants = set(frame.iloc[train_indices]["participant_id"])
    test_participants = set(frame.iloc[test_indices]["participant_id"])
    session_overlap = None if external_images else len(train_sessions & test_sessions)
    participant_overlap = None if external_images else len(train_participants & test_participants)
    source_group_overlap = None
    if external_images:
        source_group_overlap = len(set(frame.iloc[train_indices]["source_group_id"]) &
                                   set(frame.iloc[test_indices]["source_group_id"]))
        logger.warning("Filename/content group holdout does not establish unseen-session/signer performance.")
        if source_group_overlap:
            logger.warning("%s source groups occur in both splits; results may be optimistic.", source_group_overlap)
    if session_overlap:
        logger.warning("%s sessions occur in both splits: frame results may be optimistic.", session_overlap)
    if participant_overlap:
        logger.warning("%s participants occur in both splits: this is not held-out-signer evaluation.", participant_overlap)
    model = RandomForestClassifier(n_estimators=tree_count, random_state=random_state,
                                   min_samples_leaf=settings.RANDOM_FOREST_MIN_SAMPLES_LEAF, n_jobs=-1)
    logger.info("Training Random Forest: %s training rows, %s test rows.", len(train_indices), len(test_indices))
    model.fit(features[train_indices], labels[train_indices])
    predicted_labels = model.predict(features[test_indices])
    evaluation = evaluate_predictions(labels[test_indices], predicted_labels, model.classes_.tolist())
    metadata = {
        "model_type": "RandomForestClassifier", "version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "practice": practice, "dataset_path": str(dataset_path.resolve()),
        **provenance,
        "preprocessing_version": settings.PREPROCESSING_VERSION,
        "feature_count": int(model.n_features_in_), "feature_columns": FEATURE_COLUMNS,
        "labels": model.classes_.tolist(), "training_samples": len(train_indices),
        "testing_samples": len(test_indices), "test_accuracy": evaluation["accuracy"],
        "split_mode": split_mode, "requested_test_fraction": test_size,
        "actual_test_fraction": len(test_indices) / len(frame),
        "training_csv_records": (train_indices + 2).tolist(),
        "testing_csv_records": (test_indices + 2).tolist(),
        "session_overlap": session_overlap, "participant_overlap": participant_overlap,
        "source_group_overlap": source_group_overlap,
        "train_class_counts": frame.iloc[train_indices]["label"].value_counts().to_dict(),
        "test_class_counts": frame.iloc[test_indices]["label"].value_counts().to_dict(),
        "random_state": random_state, "model_parameters": model.get_params(),
        "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__, "pandas_version": pd.__version__, "joblib_version": joblib.__version__,
    }
    save_training_run(model, metadata, evaluation, output_dir)
    print("PRACTICE ONLY - metrics do not establish FSL accuracy." if practice else "Held-out evaluation")
    print(f"Training samples: {len(train_indices)} | Testing samples: {len(test_indices)}")
    print(f"Split: {split_mode} | Actual test fraction: {metadata['actual_test_fraction']:.1%}")
    print(f"Accuracy: {evaluation['accuracy']:.3f}")
    print(evaluation["classification_report_text"])
    print(f"Saved model and reports under: {output_dir.resolve()}")
    return metadata
