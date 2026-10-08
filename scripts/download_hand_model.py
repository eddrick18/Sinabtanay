"""Explicit one-time download of Google's hand-location model (no webcam input)."""

from pathlib import Path
import sys
from urllib.request import urlopen
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings


def main() -> int:
    """Save a complete model atomically; leave existing assets untouched."""
    destination = settings.HAND_MODEL_PATH
    if destination.is_file():
        print(f"Model already exists: {destination}")
        return 0
    temporary_path = destination.with_suffix(".task.tmp")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(settings.HAND_MODEL_URL, timeout=60) as response:
            model_bytes = response.read()
        if len(model_bytes) < 1_000_000:
            raise ValueError("Model response is unexpectedly small; download was not saved.")
        temporary_path.write_bytes(model_bytes)
        temporary_path.replace(destination)
        print(f"Downloaded hand-location model: {destination}")
        return 0
    except (URLError, OSError, ValueError) as error:
        print(f"ERROR: Could not download hand model: {error}")
        return 1
    finally:
        temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
