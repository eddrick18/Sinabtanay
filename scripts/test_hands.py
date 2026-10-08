"""Webcam hand detection and Phase 3 feature preview. Press Q to quit."""

import argparse
import logging
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings

try:
    import cv2
    from src.hand_tracking.hand_detector import HandDetector, draw_hand_results, get_hand_labels
    from src.hand_tracking.landmark_processor import create_feature_vector, FEATURE_COUNT
except ImportError:
    print("ERROR: Dependencies missing. Run: python -m pip install -r requirements.txt")
    raise SystemExit(1)

logger = logging.getLogger(__name__)


def run_hand_test(camera_index: int, debug: bool) -> int:
    """Connect camera and detector; return an exit status after cleanup."""
    camera = None
    detector = None
    try:
        detector = HandDetector()
        logger.info("Hand landmark model loaded.")
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            logger.error("Unable to access webcam %s. Check permissions or try another "
                         "--camera-index.", camera_index)
            return 1
        logger.info("Camera initialized. Focus the window and press Q to exit.")
        previous_status = None
        previous_feature_error = None
        if FEATURE_COUNT != settings.FEATURE_COUNT:
            raise ValueError("Configured feature count does not match preprocessing schema.")
        previous_frame_time = time.perf_counter()
        while True:
            success, frame = camera.read()
            if not success or frame is None or frame.size == 0:
                logger.error("Unable to read a webcam frame.")
                return 1
            if settings.MIRROR_CAMERA:
                frame = cv2.flip(frame, 1)
            timestamp_ms = time.monotonic_ns() // 1_000_000
            result = detector.detect(frame, timestamp_ms)
            height, width = frame.shape[:2]
            try:
                features = create_feature_vector(result, (width, height))
                feature_status = f"Features: {features.size} | usable: {bool(result.hand_landmarks)}"
                previous_feature_error = None
            except ValueError as error:
                # Tracker handedness can be ambiguous; skip this frame's features, not the webcam.
                feature_status = "Features: unusable (see terminal)"
                if str(error) != previous_feature_error:
                    logger.warning("Feature frame skipped: %s", error)
                    previous_feature_error = str(error)
            draw_hand_results(frame, result)
            labels = get_hand_labels(result)
            # Sorting avoids log changes caused only by detection-order changes.
            status = "No hand detected."
            if labels:
                status = "Hands detected: " + ", ".join(sorted(labels))
            if status != previous_status:
                logger.info(status)
                previous_status = status
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)
            cv2.putText(frame, "Hand detection only | Q: quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if debug:
                now = time.perf_counter()
                fps = 1.0 / max(now - previous_frame_time, 1e-9)
                previous_frame_time = now
                cv2.putText(frame, f"Hands: {len(labels)} | FPS: {fps:.1f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, feature_status, (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.imshow(settings.HAND_WINDOW_TITLE, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if cv2.getWindowProperty(settings.HAND_WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
        return 0
    except (FileNotFoundError, OSError, ValueError, RuntimeError, cv2.error) as error:
        logger.error("Hand detection failed: %s", error, exc_info=debug)
        return 1
    finally:
        if camera is not None:
            camera.release()
        try:
            if detector is not None:
                detector.close()
        finally:
            cv2.destroyAllWindows()
            logger.info("Camera, detector, and windows released.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=settings.CAMERA_INDEX)
    parser.add_argument("--debug", action="store_true", default=settings.DEBUG)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if arguments.debug else logging.INFO,
                        format="%(levelname)s - %(message)s")
    return run_hand_test(arguments.camera_index, arguments.debug)


if __name__ == "__main__":
    raise SystemExit(main())
