# Architecture and file guide — Phase 1 reference

Phase 2 is now implemented: see [HAND_DETECTION.md](HAND_DETECTION.md) for the
current pipeline and new modules. Phase 3 preprocessing is explained in
[LANDMARK_PROCESSING.md](LANDMARK_PROCESSING.md). Phase 4 collection is explained in
[DATA_COLLECTION.md](DATA_COLLECTION.md). The walkthrough below describes the original
webcam-only script, which remains available. Phase 5 inspection is explained in
[DATASET_VALIDATION.md](DATASET_VALIDATION.md).
[MODEL_TRAINING.md](MODEL_TRAINING.md) explains the Phase 6 pipeline and metrics.
requirements.txt now installs camera and conventional training dependencies;
settings.py includes detection, dataset and training defaults.

Current pipeline: webcam → OpenCV capture → frame + instruction overlay → OpenCV window.
No classification is implemented. Camera index selects an operating-system
device; it is not a frame number. A frame is a NumPy array of BGR pixel values.

| File | Why it exists / process | Input | Output / connection |
| --- | --- | --- | --- |
| requirements.txt | Records the one direct dependency needed now | pip reads package/version | Installs GUI-enabled OpenCV and its NumPy dependency |
| .gitignore | Keeps environments, caches, recordings, and future datasets out of Git | File paths | Git ignore rules; does not delete files |
| config/__init__.py | Marks configuration as a Python package | None | Allows config imports |
| config/settings.py | Centralizes camera index, window title, and debug default | Editable constants | Defaults used by test_camera.py |
| scripts/test_camera.py | Opens camera, reads/displays frames, handles exit and cleanup | Settings, terminal options, webcam | Live window, log messages, exit status 0 or 1 |
| README.md | Gives installation/run instructions and scope | Reader's terminal workflow | Repeatable setup and manual verification |
| docs/ARCHITECTURE.md | Explains ownership and the script line by line | Current source | Learning/reference guide |
| docs/FSL_LIMITATIONS.md | Records language, accessibility, and privacy boundaries | Project scope | Honest description of capabilities |
| docs/DEVELOPMENT_LOG.md | Records actual validation and unresolved work | Development evidence | Phase history |

## Reading test_camera.py, in source order

Each entry below explains the executable lines; blank lines separate sections.

1. The opening docstring states this script's purpose.
2. `import argparse` supplies terminal options. `import logging` supplies INFO/ERROR messages.
3. `from pathlib import Path` handles paths. `import sys` accesses Python's import search path. `import time` supplies a monotonic timer for debug FPS.
4. `PROJECT_ROOT = Path(__file__).resolve().parents[1]` finds the project folder from the script location.
5. `sys.path.insert(0, str(PROJECT_ROOT))` lets a directly launched script import the sibling config package. It does not change the working directory.
6. `from config import settings` loads central defaults.
7. `try: import cv2` loads OpenCV; the `except ImportError` block gives an installation command and exits with status 1 when it is missing.
8. `logger = logging.getLogger(__name__)` creates this module's logger.
9. `def run_camera_test(...) -> int` names the camera operation and documents its exit status. Type hints clarify expected values.
10. `camera = cv2.VideoCapture(camera_index)` asks the operating system to open that camera. Device 0 is usually the first camera; enumeration can vary.
11. `try` begins protected processing. Its `finally` always runs after entering this block, including on early returns.
12. `if not camera.isOpened()` checks whether opening succeeded. The error log provides troubleshooting; `return 1` reports failure to the terminal.
13. The INFO log announces success without printing every frame.
14. `previous_frame_time = time.perf_counter()` initializes a monotonic timer.
15. `while True` repeats capture until an exit condition or read error.
16. `success, frame = camera.read()` retrieves the next image and a success flag.
17. The next `if` rejects a failed, missing, or empty image before display. Its log and `return 1` report a camera read failure.
18. `cv2.putText(...)` modifies the frame to add instructions. `(10, 30)` is the text origin in pixels; the font constant selects a font; `0.6` sets its scale; `(0, 255, 0)` is green in BGR order; `2` sets stroke thickness.
19. `if debug` enables optional measurement. `current_frame_time` reads the timer; `elapsed` subtracts the preceding time; assigning `previous_frame_time` prepares the next iteration.
20. `fps = 1.0 / elapsed if elapsed > 0 else 0.0` estimates processed frames per second and guards division by zero. It is an instantaneous estimate, not a performance guarantee.
21. The second `putText` displays that FPS on the next line only in debug mode.
22. `cv2.imshow(settings.WINDOW_TITLE, frame)` creates/updates the named desktop window with the current image.
23. `key = cv2.waitKey(1) & 0xFF` lets OpenCV process window events and waits approximately 1 ms for a key. This is not an exact frame timer. `& 0xFF` keeps the low byte of the key code.
24. `if key in (ord("q"), ord("Q"))` compares against the numeric codes for lowercase/uppercase Q; `break` ends the capture loop. The video window must have keyboard focus.
25. `cv2.getWindowProperty(..., cv2.WND_PROP_VISIBLE) < 1` checks whether the user closed the window; its `break` also exits normally.
26. `return 0` indicates a successful user-controlled exit.
27. `except cv2.error` catches OpenCV camera/display errors. The log's `exc_info=debug` adds traceback details only in debug mode. `return 1` indicates failure.
28. `finally` performs cleanup even after read errors or keyboard interruption. `camera.release()` relinquishes the webcam. `cv2.destroyAllWindows()` closes OpenCV windows. The final log confirms cleanup.
29. `def main() -> int` separates terminal setup from the capture loop.
30. `ArgumentParser(description=__doc__)` creates a parser with the script docstring as help text.
31. `add_argument("--camera-index", ...)` accepts an integer device index and uses settings as its default. Its `help` text explains the option.
32. `add_argument("--debug", action="store_true", ...)` makes a flag that enables debug mode; its default also comes from settings.
33. `arguments = parser.parse_args()` validates terminal options. `--help` prints help and exits before opening a camera.
34. `logging.basicConfig(...)` chooses the log level and a compact severity/message format.
35. `return run_camera_test(...)` passes the parsed options into the camera function and forwards its status.
36. `if __name__ == "__main__"` runs the entry point only when executed directly. `raise SystemExit(main())` returns its status to Windows.

## Future architecture (planned, not implemented)

OpenCV gets/displays frames. MediaPipe will locate hands and landmarks. A shared
landmark processor will normalize coordinates for both collection and inference.
A conventional classifier will predict trained labels; a smoother will reduce
flicker; application logic will construct text and optional speech. Add focused
modules when their phases begin, without combining those responsibilities.

Detection, preprocessing, controlled collection, inspection, baseline training,
held-out metrics and model saving are implemented. Dedicated evaluation and runtime
model loading/recognition remain later phases. Real training needs verified data.

## External alphabet images
The image import adapter uses HandDetector in IMAGE mode and the existing feature processor. It writes an external-images schema with file/hash/group provenance and UNKNOWN recording identities. See [IMAGE_IMPORT.md](IMAGE_IMPORT.md) for filtering, output files and evaluation limits.

## Live model runtime
scripts/run_live.py connects the webcam and shared landmark processor to src/recognition/predictor.py. The predictor checks the saved model contract and smooths threshold-qualified class scores. See LIVE_RECOGNITION.md.

## Message buffer
MessageBuffer owns the text and explicit keyboard editing. The live loop supplies only the current stable prediction, renders a footer, and saves no message data. See MESSAGE_BUFFER.md.

## Desktop application
scripts/run_app.py launches src/ui/app.py. Tk scheduled frame updates reuse LivePredictor, MessageBuffer and LocalSpeech; the original OpenCV preview remains available. See DESKTOP_UI.md for layout and checks.
