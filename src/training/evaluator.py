"""Metrics and confusion-matrix rendering; no camera or training here."""

import os
from pathlib import Path

from config import settings

os.environ.setdefault("MPLCONFIGDIR", str(settings.PROJECT_ROOT / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")  # Save plots without opening another GUI window.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


def evaluate_predictions(actual_labels, predicted_labels, labels: list[str]) -> dict:
    """Compute held-out results with one consistent label order in every output."""
    return {
        "accuracy": float(accuracy_score(actual_labels, predicted_labels)),
        "classification_report": classification_report(actual_labels, predicted_labels,
                                                       labels=labels, output_dict=True, zero_division=0),
        "classification_report_text": classification_report(actual_labels, predicted_labels,
                                                            labels=labels, digits=3, zero_division=0),
        "confusion_matrix": confusion_matrix(actual_labels, predicted_labels, labels=labels).tolist(),
        "labels": labels,
    }


def save_confusion_matrix(evaluation: dict, path: Path, practice: bool) -> None:
    """Save counts: rows are actual classes; columns are predicted classes."""
    figure, axis = plt.subplots(figsize=(max(6, len(evaluation["labels"]) * 0.9), 5.5))
    try:
        display = ConfusionMatrixDisplay(confusion_matrix=np.asarray(evaluation["confusion_matrix"]),
                                         display_labels=evaluation["labels"])
        display.plot(ax=axis, cmap="Blues", colorbar=False, xticks_rotation=35, values_format="d")
        axis.set_title("Practice only - not FSL accuracy" if practice else "Held-out sign classification")
        figure.tight_layout()
        figure.savefig(path, dpi=150)
    finally:
        plt.close(figure)
