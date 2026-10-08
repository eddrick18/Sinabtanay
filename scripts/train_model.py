"""Phase 6: validate, split, train Random Forest, evaluate and save outputs."""

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings

try:
    from src.training.trainer import train_model
except ImportError:
    print("ERROR: Training dependencies missing. Run: python -m pip install -r requirements.txt")
    raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=settings.DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, help="New model/report directory; existing runs are preserved.")
    parser.add_argument("--split", choices=["auto", "session", "group", "frame"], default="auto",
                        help="Auto: session holdout for webcam data; source-group holdout for imported images.")
    parser.add_argument("--test-size", type=float, default=settings.TEST_SIZE)
    parser.add_argument("--trees", type=int, default=settings.RANDOM_FOREST_TREES)
    parser.add_argument("--practice", action="store_true", help="Explicit unverified experiment; outputs default to models/practice.")
    parser.add_argument("--debug", action="store_true", default=settings.DEBUG)
    arguments = parser.parse_args()
    if not 0 < arguments.test_size < 1 or arguments.trees < 1:
        parser.error("Test size must be in (0,1); tree count must be positive.")
    output_dir = arguments.output_dir
    if output_dir is None:
        output_dir = settings.PROJECT_ROOT / "models"
        if arguments.practice:
            output_dir /= "practice"
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    try:
        train_model(arguments.dataset, output_dir, split_mode=arguments.split,
                    test_size=arguments.test_size, tree_count=arguments.trees, practice=arguments.practice)
        return 0
    except FileNotFoundError:
        logging.error("Dataset not found: %s. Collect verified examples with scripts/collect_data.py first.", arguments.dataset)
        return 1
    except (OSError, ValueError, RuntimeError) as error:
        logging.error("Training failed: %s", error, exc_info=arguments.debug)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
