# Phase 2 — understanding hand detection

Pipeline: webcam → OpenCV BGR frame → optional horizontal mirror → RGB image →
MediaPipe Hand Landmarker → landmarks and handedness → OpenCV drawing/display.
All inference is local. Nothing is recorded or uploaded. Downloading the model
is a separate one-time step that sends no webcam input.

## What MediaPipe does

**Detection** locates a hand region in an image. **Tracking** follows that region
between frames, avoiding a full palm-detection pass when tracking succeeds.
**Sign recognition** would assign a trained vocabulary label to hand features;
it is not part of this phase. The system knows where hands are, not what sign
they represent.

Google's pretrained model bundle contains palm detection and hand-landmark
estimation models. It returns 21 landmarks per hand, image coordinates, world
coordinates, and handedness. We use image coordinates for drawing and the
handedness category for labels. No FSL model is trained or loaded.

We use the Tasks API, with VIDEO mode for sequential webcam frames. This blocks
until each frame's result is ready, keeping results aligned with the displayed
frame and avoiding callbacks/threads. It still uses temporal tracking. LIVE_STREAM
mode is a possible later optimization, not required for this phase.

Source: [Google's Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)
and [model overview](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/index).

## Files added and changed

| File | Purpose and input | Output and connection |
| --- | --- | --- |
| src/__init__.py, src/hand_tracking/__init__.py | Define importable application packages; no runtime inputs | Organize the detector module |
| src/hand_tracking/hand_detector.py | BGR frame, monotonic timestamp, settings, downloaded model | MediaPipe result; drawing functions modify the displayed frame |
| scripts/test_hands.py | Webcam, settings, --camera-index and --debug | Annotated live window, status logs and exit code; owns camera/detector cleanup |
| scripts/download_hand_model.py | Configured version-1 Google model URL | models/pretrained/hand_landmarker.task; reused on subsequent runs |
| models/pretrained/hand_landmarker.task | Google's pretrained binary, not our FSL classifier | Loaded by HandDetector; ignored by Git and downloadable again |
| tests/test_hand_detector.py | Blank images and synthetic drawing results | Three camera-free checks via standard-library unittest |
| config/settings.py | Camera defaults and detector thresholds, model path/URL, mirror setting | Central configuration consumed by scripts and detector |
| requirements.txt | Pinned MediaPipe and GUI-enabled OpenCV contrib versions | Reproducible direct dependency installation |
| .gitignore | Cache/model paths | Excludes generated cache and downloaded binary |
| README.md, docs/ARCHITECTURE.md, docs/FSL_LIMITATIONS.md | Current implementation status | Updated setup and scope |
| docs/HAND_DETECTION.md | This learning guide | Explanation and manual acceptance checklist |
| docs/DEVELOPMENT_LOG.md | Execution evidence | Records successes and remaining hardware checks |

## Reading the important logic

`HandDetector.__init__` checks the model path, constructs options from settings,
and creates one tracker. Keeping that tracker alive across frames enables
tracking; recreating it every frame would waste work and reset state.

`detect(frame, timestamp_ms)` guards against equal timestamps, converts BGR to
RGB with `cv2.cvtColor`, wraps the RGB array in `mp.Image`, and calls
`detect_for_video`. The timestamp uses monotonic milliseconds rather than the
wall clock, which can change. It returns the original MediaPipe result so Phase 3
can later receive landmarks without duplicating detection.

`get_hand_labels` extracts one category name per detected hand and uses Unknown
when handedness is unavailable. Unknown here means unknown handedness, not an
FSL prediction. Detection order is not a persistent hand identity.

`draw_hand_results` converts normalized x/y to pixel positions using image width
and height. It draws skeleton edges with `cv2.line`, joints with `cv2.circle`,
and a label near each wrist with `cv2.putText`. This coordinate conversion only
supports display; wrist/scale normalization for ML belongs to Phase 3.

`run_hand_test` creates the detector once, opens the camera, and loops over frames.
Mirroring happens before both detection and drawing so they share coordinates.
Each iteration detects, draws, updates status, displays, then checks Q/window
closure. No-hand results produce a normal message. Logs change only when status
changes, instead of printing on every frame. Debug mode adds approximate FPS.
The `finally` block releases the camera, closes the detector, and closes windows.

`download_hand_model.py` downloads only on explicit execution and only when the
model is missing. It writes to a temporary file, then renames it after a complete
download so interrupted downloads do not become the final asset. Its size check
is a basic error check, not cryptographic verification. It does not run during
webcam inference. If a model is corrupted, delete that .task file and rerun it.

## Settings and interpretation

MAX_HANDS=2 limits detections. Detection, hand-presence, and tracking thresholds
are 0.6. These are hand-pipeline thresholds, not FSL confidence or accuracy.
MIRROR_CAMERA=True gives a familiar selfie view. Check handedness with your known
physical left/right hand, particularly if a virtual camera already mirrors its
output; a second mirror can reverse the expected convention. Adjust the central
mirror setting if needed, then keep that convention consistent in later phases.

MediaPipe requires opencv-contrib-python and imports matplotlib as a dependency.
We replaced opencv-python with the GUI-enabled contrib variant rather than
installing both packages that own cv2. No web UI, audio capture, sign classifier,
or TensorFlow training was added. The internal TensorFlow Lite/XNNPACK startup
message comes from MediaPipe's own pretrained-model runtime.

## Manual acceptance before Phase 3

1. Start test_hands.py. With hands out of view, check “No hand detected.”
2. Show one hand fully in view: check 21 landmark dots, connections, and label.
3. Test your physical left hand and right hand separately; verify labels.
4. Show both hands with some separation: check two skeletons and both labels.
5. Move hands, remove them, and return them: status should update without crashing.
6. Press Q with window focused; check normal exit and camera release.

If detection is unreliable, improve front lighting, keep the whole hand visible,
avoid overlap, and close other camera apps. This phase does not validate sign
recognition or guarantee detection on every pose. Camera-free tests cannot prove
live hand detection works on your webcam.
