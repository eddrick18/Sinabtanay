"""Phase 1: show a local webcam feed until Q or the close button is pressed."""

import argparse
import logging
from pathlib import Path
import sys
import time

# Direct script execution adds scripts/, not the project root, to Python's path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings

try:
    import cv2
except ImportError:
    print("ERROR: OpenCV is missing. Run: python -m pip install -r requirements.txt")
    raise SystemExit(1)

logger = logging.getLogger(__name__)


def run_camera_test(camera_index: int, debug: bool = False) -> int:
    """Display frames; return 0 on normal exit or 1 on a camera/display error."""
    camera = cv2.VideoCapture(camera_index)
    try:
        if not camera.isOpened():
            logger.error("Unable to access webcam %s. Check Windows camera permissions, "
                         "close other camera apps, or try another --camera-index.", camera_index)
            return 1

        logger.info("Camera initialized. Focus the video window and press Q to exit.")
        previous_frame_time = time.perf_counter()

        while True:
            success, frame = camera.read()
            if not success or frame is None or frame.size == 0:
                logger.error("Unable to read a webcam frame. Check the camera connection.")
                return 1

            cv2.putText(frame, "SignBridge FSL | Webcam test | Q: quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if debug:
                current_frame_time = time.perf_counter()
                elapsed = current_frame_time - previous_frame_time
                previous_frame_time = current_frame_time
                fps = 1.0 / elapsed if elapsed > 0 else 0.0
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow(settings.WINDOW_TITLE, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if cv2.getWindowProperty(settings.WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break

        return 0
    except cv2.error:
        logger.error("OpenCV camera/display operation failed. Use a local Windows "
                     "desktop and the GUI-enabled opencv-python package.", exc_info=debug)
        return 1
    finally:
        camera.release()
        cv2.destroyAllWindows()
        logger.info("Camera released and windows closed.")


def main() -> int:
    """Read terminal options, configure logging, and start the webcam test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=settings.CAMERA_INDEX,
                        help="Camera device index (default: 0).")
    parser.add_argument("--debug", action="store_true", default=settings.DEBUG,
                        help="Show approximate FPS and OpenCV error details.")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if arguments.debug else logging.INFO,
                        format="%(levelname)s - %(message)s")
    return run_camera_test(arguments.camera_index, arguments.debug)


if __name__ == "__main__":
    raise SystemExit(main())
