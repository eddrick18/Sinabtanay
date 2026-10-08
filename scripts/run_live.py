"""Live static alphabet prediction. Press Q to quit. No frames are saved."""
import argparse
import logging
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings
from src.hand_tracking.hand_detector import HandDetector, draw_hand_results
from src.hand_tracking.landmark_processor import create_feature_vector
from src.recognition.predictor import LivePredictor
from src.recognition.message_buffer import MessageBuffer
from src.speech.local_speech import LocalSpeech
import numpy as np
import cv2


def run_live(model_dir, camera_index, threshold, window, votes, debug=False, speech_enabled=True):
    camera = detector = None
    speech = LocalSpeech(speech_enabled)
    try:
        predictor = LivePredictor(model_dir, threshold, window, votes)
        message = MessageBuffer(predictor.classes)
        message_status = 'Ready - add letters with Enter'
        logging.info('Loaded %s classes from %s', len(predictor.classes), model_dir)
        detector = HandDetector()
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            raise RuntimeError('Cannot open webcam. Close other camera apps or try --camera-index 1.')
        title = 'Sinabtanay - Live Alphabet'
        last_timestamp = -1
        last_time = time.perf_counter()
        while True:
            ok, frame = camera.read()
            if not ok or frame is None or not frame.size:
                raise RuntimeError('Cannot read webcam frame.')
            if settings.MIRROR_CAMERA:
                frame = cv2.flip(frame, 1)
            timestamp = max(time.monotonic_ns() // 1_000_000, last_timestamp + 1)
            last_timestamp = timestamp
            result = detector.detect(frame, timestamp)
            label, confidence = None, None
            if not result.hand_landmarks:
                predictor.reset()
                status = 'No hand detected'
            else:
                height, width = frame.shape[:2]
                try:
                    features = create_feature_vector(result, (width, height))
                except ValueError:
                    predictor.reset()
                    status = 'Ambiguous hand - reposition'
                else:
                    label, confidence, status = predictor.predict(features)
            draw_hand_results(frame, result)
            lines = [f'Letter: {label or "--"}', status,
                     f'Model score: {confidence:.0%}' if confidence is not None else 'Model score: --',
                     f'Hands detected: {len(result.hand_landmarks)}',
                     'Practice alphabet | J/Z excluded | Q: quit']
            if not predictor.metadata.get('practice'):
                lines[-1] = 'Static signs | Q: quit'
            now = time.perf_counter()
            if debug:
                lines.append(f'FPS: {1 / max(now-last_time, 1e-9):.1f} | threshold: {threshold:.0%}')
            last_time = now
            # Solid panel keeps labels readable against any camera background.
            panel_height = min(frame.shape[0], 30 * len(lines) + 12)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], panel_height), (25, 25, 25), -1)
            for i, text in enumerate(lines):
                cv2.putText(frame, text, (10, 27 + 30*i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (80, 255, 100) if i == 0 else (255, 255, 255), 2)
            footer = np.full((165, frame.shape[1], 3), 25, dtype=np.uint8)
            # Show the end of long messages; edits still apply to the full buffer.
            visible = message.text or '(empty)'
            available = max(frame.shape[1] - 20, 1)
            while cv2.getTextSize('Message: ' + visible, cv2.FONT_HERSHEY_SIMPLEX, .6, 1)[0][0] > available and len(visible) > 1:
                visible = visible[1:]
            if message.text and visible != message.text:
                visible = '...' + visible[3:]
            footer_lines = ['Message: ' + visible,
                            'Enter: add | Space: word space',
                            'Backspace: delete | C: clear | Q: quit',
                            f'{message_status} | {len(message.text)}/{message.limit}',
                            'T: speak | Esc: stop | ' + speech.poll()]
            for i, text in enumerate(footer_lines):
                text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .6, 1)[0][0]
                scale = .6 * min(1, available / max(text_width, 1))
                cv2.putText(footer, text, (10, 27 + 30*i), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, (255, 255, 255), 1)
            cv2.imshow(title, np.vstack((frame, footer)))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            if key in (ord('t'), ord('T')):
                message_status = speech.speak(message.text)
            elif key == 27:
                message_status = speech.stop()
            update = message.handle_key(key, label)
            if update is not None:
                message_status = update
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, AttributeError, cv2.error) as error:
        logging.error('Live prediction failed: %s', error, exc_info=debug)
        return 1
    finally:
        speech.close()
        if camera is not None:
            camera.release()
        try:
            if detector is not None:
                detector.close()
        finally:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=settings.PROJECT_ROOT / 'models' / 'alphabet_single_hand_v3')
    parser.add_argument('--camera-index', type=int, default=settings.CAMERA_INDEX)
    parser.add_argument('--threshold', type=float, default=0.70)
    parser.add_argument('--window', type=int, default=7)
    parser.add_argument('--votes', type=int, default=5)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no-speech', action='store_true', help='Disable optional local text-to-speech.')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    return run_live(args.model_dir, args.camera_index, args.threshold, args.window, args.votes, args.debug, not args.no_speech)


if __name__ == '__main__':
    raise SystemExit(main())
