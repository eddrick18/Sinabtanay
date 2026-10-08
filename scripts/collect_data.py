"""Phase 4: press Space to save one landmark example; Q quits."""

import argparse
import logging
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings

try:
    import cv2
    import numpy as np
    from src.dataset.collector import SampleCollector
    from src.dataset.capture_timer import CaptureTimer
    from src.dataset.dataset_utils import DatasetWriter
    from src.hand_tracking.hand_detector import HandDetector, draw_hand_results, get_hand_labels
except ImportError:
    print("ERROR: Dependencies missing. Run: python -m pip install -r requirements.txt")
    raise SystemExit(1)

logger = logging.getLogger(__name__)


def run_collection(arguments: argparse.Namespace) -> int:
    """Own the webcam loop; collection/preprocessing/storage stay in their modules."""
    camera = None
    detector = None
    try:
        writer = DatasetWriter(arguments.dataset, getattr(arguments, "vocabulary", "words"))
        collector = SampleCollector(writer, arguments.label, arguments.participant_id,
                                    uuid.uuid4().hex, arguments.reference, arguments.target)
        timed = getattr(arguments, "timed", False)
        timer = CaptureTimer(getattr(arguments, "interval", 0.75), getattr(arguments, "countdown", 3.0))
        logger.info("Session %s | participant %s | dataset %s", collector.session_id,
                    collector.participant_id, writer.path)
        if arguments.reference == "UNVERIFIED":
            logger.warning("Reference is UNVERIFIED. These are practice examples, not verified FSL data.")
        detector = HandDetector()
        camera = cv2.VideoCapture(arguments.camera_index)
        if not camera.isOpened():
            logger.error("Unable to access webcam. Check permissions or another --camera-index.")
            return 1
        capture_status = "Ready: tap SPACE to capture"
        previous_frame_time = time.perf_counter()
        while True:
            success, frame = camera.read()
            if not success or frame is None or frame.size == 0:
                logger.error("Unable to read webcam frame.")
                return 1
            if settings.MIRROR_CAMERA:
                frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            result = detector.detect(frame, time.monotonic_ns() // 1_000_000)
            now_capture = time.monotonic() if timed else 0.0
            if timed and timer.due(now_capture):
                capture_status = collector.capture(result, (width, height), now_capture)
                if collector.session_count >= collector.target:
                    timer.pause()
                    capture_status = "Target reached - Q: quit"
            draw_hand_results(frame, result)
            labels = get_hand_labels(result)
            lines = [f"Label: {arguments.label}",
                     f"Session: {collector.session_count}/{collector.target} | Label total: "
                     f"{writer.sample_counts.get(arguments.label, 0)}",
                     "Hands: " + (", ".join(labels) if labels else "none"),
                     timer.status(now_capture) if timed else "SPACE: capture | Q: quit", capture_status,
                     "Reference: " + arguments.reference[:55]]
            # Reserve a solid header for readable status without hiding hand landmarks.
            panel = np.zeros((205, width, 3), dtype=np.uint8)
            for index, line in enumerate(lines):
                text_width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
                text_scale = 0.55 * min(1.0, max(width - 20, 1) / max(text_width, 1))
                cv2.putText(panel, line, (10, 25 + index * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), 1)
            if arguments.debug:
                now = time.perf_counter()
                fps = 1.0 / max(now - previous_frame_time, 1e-9)
                previous_frame_time = now
                cv2.putText(panel, f"FPS: {fps:.1f} | Features: {settings.FEATURE_COUNT}",
                            (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow(settings.COLLECTION_WINDOW_TITLE, np.vstack((panel, frame)))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if cv2.getWindowProperty(settings.COLLECTION_WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
            if timed and key in (ord("s"), ord("S")) and collector.session_count < collector.target:
                timer.toggle(time.monotonic())
            if not timed and key == ord(" "):
                capture_status = collector.capture(result, (width, height), time.monotonic())
                logger.info(capture_status)
        return 0
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        logger.error("Collection failed: %s", error, exc_info=arguments.debug)
        return 1
    finally:
        if camera is not None:
            camera.release()
        try:
            if detector is not None:
                detector.close()
        finally:
            cv2.destroyAllWindows()
            logger.info("Collection resources released.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocabulary", choices=["words", "alphabet"], default="words")
    parser.add_argument("--label", required=True)
    parser.add_argument("--participant-id", default="signer_01", help="Consistent pseudonym, not a real name.")
    parser.add_argument("--reference", default="UNVERIFIED", help="Verified FSL source/consultation note, or UNVERIFIED for practice.")
    parser.add_argument("--target", type=int, default=settings.COLLECTION_TARGET)
    parser.add_argument("--timed", action="store_true", help="S starts/pauses interval capture after a countdown.")
    parser.add_argument("--interval", type=float, default=0.75)
    parser.add_argument("--countdown", type=float, default=3.0)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--camera-index", type=int, default=settings.CAMERA_INDEX)
    parser.add_argument("--debug", action="store_true", default=settings.DEBUG)
    arguments = parser.parse_args()
    try:
        CaptureTimer(arguments.interval, arguments.countdown)
    except ValueError as error:
        parser.error(str(error))
    labels = settings.ALPHABET_LABELS if arguments.vocabulary == "alphabet" else settings.SIGN_LABELS
    if arguments.label not in labels:
        parser.error("Choose a label from: " + ", ".join(labels))
    if arguments.dataset is None:
        arguments.dataset = (settings.PROJECT_ROOT / "data/processed/alphabet_webcam.csv"
                             if arguments.vocabulary == "alphabet" else settings.DATASET_PATH)
    if arguments.target <= 0 or not arguments.participant_id.strip() or not arguments.reference.strip():
        parser.error("Target must be positive and participant/reference must be nonempty.")
    logging.basicConfig(level=logging.DEBUG if arguments.debug else logging.INFO,
                        format="%(levelname)s - %(message)s")
    return run_collection(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
