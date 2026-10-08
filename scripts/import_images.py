"""Convert the extracted alphabet images to a separate audited landmark CSV."""

import argparse
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings

try:
    from src.dataset.image_importer import import_images
except ImportError:
    print("ERROR: Dependencies missing. Run: python -m pip install -r requirements.txt")
    raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=settings.ALPHABET_SOURCE_PATH)
    parser.add_argument("--output", type=Path, default=settings.ALPHABET_DATASET_PATH)
    parser.add_argument("--debug", action="store_true", default=settings.DEBUG)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    try:
        report = import_images(arguments.source, arguments.output)
        print(json.dumps(report, indent=2))
        return 0 if report["retained_samples"] else 1
    except (OSError, ValueError, RuntimeError) as error:
        logging.error("Image import failed: %s", error, exc_info=arguments.debug)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
