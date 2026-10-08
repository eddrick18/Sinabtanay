# Phase 3 — turning landmarks into features

The classifier we will train later needs numbers in a consistent order. This
phase adds that representation, not a classifier or dataset collection.

Pipeline: MediaPipe result + frame width/height → extraction → wrist translation
→ aspect correction → scale normalization → flattening → Left/Right slots.

## Files and responsibilities

| File | Purpose / inputs | Outputs / connections |
| --- | --- | --- |
| src/hand_tracking/landmark_processor.py | Pure NumPy functions; input is MediaPipe-like landmarks, handedness and (width, height) | One 126-value float32 vector, or ValueError for unusable input; called by test_hands.py and later shared by collection/inference |
| tests/test_landmark_processor.py | Synthetic coordinates, malformed results, translations and scales | 12 numerical tests; no camera or MediaPipe import required |
| scripts/test_hands.py (updated) | Camera frames, detector results and settings | Existing skeleton view plus optional feature length/usable status; malformed feature frames are skipped without stopping capture |
| config/settings.py (updated) | PREPROCESSING_VERSION and expected FEATURE_COUNT | Version/count to record in future datasets and model metadata |
| README.md and docs (updated) | Current phase and validation evidence | Run instructions, limitations and development history |

No dependencies were added. Unit tests use the standard-library unittest runner.

## How the numerical functions work

`extract_hand_landmarks` reads x, y and z from each of 21 landmark objects and
creates a (21, 3) float64 array for intermediate calculations. The shape means
21 rows (joints), with 3 columns per row (coordinates). It rejects missing joints
and missing/nonnumerical coordinates. `validate_coordinates` checks the shape
and rejects NaN or infinity, which could corrupt downstream model inputs.

`normalize_landmarks` subtracts row 0, the wrist, from every row. For example,
a wrist at x=0.4 and a fingertip at x=0.6 become a relative fingertip x=0.2.
Moving the whole hand to the right no longer changes that relative coordinate.

Image x is normalized by width, image y by height, and image z is an estimated
depth on roughly the x scale. See [MediaPipe's coordinate documentation](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python).
We multiply wrist-relative y by height/width so x and y correspond to comparable
image-width units. This corrects the aspect ratio without treating z as meters.
It does not compensate for stretching a video or reconstruct calibrated 3D data.

Next, calculate each joint's distance from the wrist:

```text
relative_i = landmark_i - wrist
relative_i.y *= height / width
radius_i = sqrt(relative_i.x² + relative_i.y² + relative_i.z²)
scale = maximum radius among the 21 joints
normalized_i = relative_i / scale
```

The furthest joint has radius 1 after normalization. A valid hand with uniform
coordinate scaling gives the same representation. This reduces camera-distance
effects but cannot guarantee identical values under perspective, occlusion, or
tracker noise. If all joints collapse onto the wrist or the scale is too small,
the function raises ValueError instead of dividing by zero. It leaves its input
array unchanged. Rotation is preserved; we do not rotate or reflect hands into
a canonical pose. A bent hand can change which joint determines the scale.

`flatten_landmarks` converts the normalized array to 63 float32 values in order:
x0,y0,z0,x1,y1,z1,...,x20,y20,z20. The first three values are zero because the
wrist is now the origin. float32 keeps the feature storage compact.

`create_feature_vector` starts with a zero-filled 126-value array. It identifies
the hand's slot using handedness, then calls extraction, normalization and
flattening. It inserts the result in the proper slot. It never sorts by horizontal
image position, so crossing the hands does not intentionally exchange slots.
MediaPipe handedness can still make mistakes; there is no identity tracker here.

```text
Python indices 0..62   = Left hand
Python indices 63..125 = Right hand
```

A missing slot stays zero. No detected hands returns 126 zeros for testing and
display, but this is NOT an example to save or classify. The caller must check
the original result contains a hand. Unknown/missing handedness, duplicate
Left/Right labels, excess hands, invalid dimensions or invalid coordinates are
rejected. Guessing a slot would introduce inconsistent training data.

The processor uses the actual frame dimensions, not hardcoded resolution. The
hand layout always remains 126 values even if MAX_HANDS is reduced to 1. Changing
that detector setting controls detection capacity, not the feature schema.

## Information preserved and lost

This baseline retains finger geometry, orientation relative to the camera,
handedness slots, and whether each slot is populated. It loses absolute wrist
location, apparent hand size, relative wrist separation, and relative size of
the two hands because each hand is normalized independently. It also contains
no movement history, face, pose or grammar.

Consequently, some different signs may become indistinguishable. Select verified
static signs whose distinguishing cues fit this representation. Do not assume
HELLO, THANK_YOU or any other proposed label is static or separable. We should
review vocabulary suitability with FSL collaborators before collecting examples.
Later location or inter-hand features require a documented new schema/version,
fresh compatible data and retraining, rather than silently changing 126 inputs.

PREPROCESSING_VERSION is currently `wrist-max-radius-aspect-v1`. Future collectors
must save this version, feature order/count, frame/mirror convention and data
provenance. Training consumes those saved features; collection and inference must
call this same processor rather than implementing their own normalization.

## Run and check

From the project folder in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\test_hands.py --debug
```

The first command runs numerical and detector checks. The second runs the live
preview with FPS, hand count, feature count and usability. No-hand frames should
show `Features: 126 | usable: False`; a valid one/two-hand result should show
`Features: 126 | usable: True`. Ambiguous results show an unusable message and
an explanatory terminal warning; the camera keeps running. Press Q to exit.

The tests establish mathematical translation/scale invariance, aspect correction,
fixed size, missing-hand padding, order consistency and rejection of invalid
inputs. Synthetic landmarks do not prove sign recognition or real-world accuracy.
After confirming the preview, Phase 4 will add deliberate, label-controlled
capture with no-hand rejection. No samples or footage are saved in Phase 3.
