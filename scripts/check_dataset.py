"""Phase 5: inspect numerical samples without changing the dataset."""

import argparse
import csv
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings

try:
    from src.dataset.validator import format_report, validate_dataset
except ImportError:
    print("ERROR: Dependencies missing. Run: python -m pip install -r requirements.txt")
    raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=settings.DATASET_PATH)
    parser.add_argument("--min-samples", type=int, default=settings.MIN_SAMPLES_PER_CLASS,
                        help="Advisory per-class minimum, not a guarantee of quality.")
    parser.add_argument("--imbalance-ratio", type=float, default=settings.CLASS_IMBALANCE_RATIO)
    parser.add_argument("--require-training-ready", action="store_true",
                        help="Return status 2 when basic training checks are not met.")
    parser.add_argument("--debug", action="store_true", default=settings.DEBUG)
    arguments = parser.parse_args()
    if arguments.min_samples < 1 or not 0 < arguments.imbalance_ratio <= 1:
        parser.error("Minimum samples must be positive; imbalance ratio must be in (0, 1].")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    try:
        report = validate_dataset(arguments.dataset, arguments.min_samples, arguments.imbalance_ratio)
        print(format_report(report))
        if report.errors:
            return 1
        if arguments.require_training_ready and not report.passes_basic_training_checks:
            return 2
        return 0
    except FileNotFoundError:
        logging.error("Dataset not found: %s. Collect samples with python scripts/collect_data.py "
                      "--label <configured-label>, or select your practice CSV using --dataset.",
                      arguments.dataset)
        return 1
    except (OSError, UnicodeError, ValueError, csv.Error) as error:
        logging.error("Dataset could not be checked: %s", error, exc_info=arguments.debug)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
