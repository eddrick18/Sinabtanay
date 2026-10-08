# Development log

## 2026-10-07 — Phase 1

- Inspected the workspace: only outputs/ and work/ existed; no existing project code.
- Verified python and py: Python 3.14.6. Verified pip: 26.1.2.
- Created minimal OpenCV camera loop, central settings, Windows setup instructions,
  privacy/FSL limitations, and a source walkthrough. No MediaPipe or ML added.
- Created .venv using the verified interpreter. Initial pip installation failed
  with WinError 10013 because session network access was restricted. Requested
  network permission, then successfully installed opencv-python 4.14.0.94 and
  NumPy 2.5.3. No global packages were changed.
- Passed Python compileall syntax validation, script --help, and pip check.
- Attempted the real camera function with a validation wrapper that would quit
  after 30 displayed frames. Device 0 could not open; zero frames were displayed.
  The function returned status 1 and executed resource cleanup. Tried device 1
  through the normal script: also unavailable with cleanup and status 1.
- Windows camera-device inventory via Win32_PnPEntity was denied by the session.
  The evidence cannot distinguish missing hardware from permission/session
  access restrictions. Live display and physical Q-key exit remain unverified.
- No automated camera substitute was used to claim hardware success. Webcam
  acceptance is a manual check; no ML or landmark tests belong to this phase.
- Remaining: user confirmation of a moving webcam image and normal Q exit before Phase 2.

## 2026-10-07 — Phase 2

- User supplied a screenshot showing successful webcam display, then explicitly
  requested Phase 2. Physical Q-key acceptance was not separately reported.
- Added MediaPipe Tasks Hand Landmarker in synchronous VIDEO mode, up to two hands,
  skeleton drawing, handedness labels, no-hand status, mirror setting, debug FPS,
  and logging on status changes. No FSL classifier or Phase 3 preprocessing added.
- Installed MediaPipe 1.1.0 for existing Python 3.14.6. Its dependency initially
  installed an additional OpenCV variant. Removed both OpenCV packages and
  installed only opencv-contrib-python 4.14.0.94; pinned it in requirements.
- Downloaded Google's version-1 hand_landmarker.task and added an explicit
  repeatable download script. Webcam inference does not perform network requests.
- MediaPipe imported matplotlib and tried to write a cache in the home folder.
  Set its default cache path to project .cache/matplotlib; subsequent checks passed
  without that permission warning. Documented internal Lite runtime messages.
- Passed three camera-free unittest checks: real model no-hand inference on blank
  frames with timestamp guarding; synthetic two-hand rendering/labels; missing
  handedness fallback. Synthetic rendering does not establish detection accuracy.
- Passed compileall, command-line help, and pip check. Real webcam test loaded the
  model but camera 0 remained unavailable in this tool session. It returned status
  1 and cleaned up camera, tracker, and windows.
- Remaining: user's manual confirmation of live one/two-hand detection, physical
  handedness labels, and Q exit before implementing Phase 3.

## 2026-10-07 — Phase 3

- User confirmed Phase 2 was done and asked to proceed.
- Added focused extraction, validation, wrist translation, image-aspect correction,
  maximum-radius scaling, flattening and fixed Left/Right slots (126 float32 values).
  Added preprocessing version/count settings; no dataset or classifier created.
- No-hand results produce zero padding but are explicitly ineligible for training.
  Missing/unknown or duplicate handedness is rejected rather than guessed.
- Updated existing hand preview to process features and show length/usability only
  in debug mode. Invalid feature frames report a warning and do not stop capture.
- Added 12 numerical tests. All 15 tests including Phase 2 checks passed, along
  with compileall and CLI help. No new dependency installation was required.
- Documented loss of location/inter-hand information and implications for verified
  vocabulary selection. This representation is a hand-shape baseline, not a full
  FSL encoding. Changes to it require a new preprocessing version and retraining.
- Remaining: user confirmation of live debug feature preview before Phase 4.

## 2026-10-07 — Phase 4

- User confirmed the feature preview works; proceeded to controlled collection.
- Added Space-triggered capture, half-second cooldown, configurable session target,
  label/session counts, hand/capture status and readable separate status panel.
- Kept detection and preprocessing in their existing modules. Added focused
  SampleCollector policy and DatasetWriter persistence modules using standard CSV,
  avoiding a Pandas dependency before dataset analysis is needed.
- Added six placeholder labels with an explicit verification comment. Reference
  defaults to UNVERIFIED, enabling a separate practice test without claiming FSL
  verification. Actual gestures must be verified before collecting training data.
- Each saved row contains 126 features plus label, participant pseudonym, random
  session ID, UTC timestamp, hand count, image size and source note. Context is
  for provenance/grouped evaluation; it must not become classifier input.
- Added schema metadata checks, exact rounded duplicate rejection across restarts,
  contradictory-label rejection, and no-hand/ambiguous-feature skips. Full dataset
  summary/imbalance reporting remains Phase 5. Concurrent writers are unsupported.
- All 24 tests passed, including nine new collection checks and a simulated
  webcam/key-event test confirming Space-only save, no-hand rejection and cleanup.
  Also passed CLI help and compileall. Tests used temporary data; no real samples
  or footage were collected. Physical collector controls still require user check.
- Remaining: confirm the five-sample practice collector and inspect local CSV
  before Phase 5. Review verified vocabulary suitability before real FSL collection.

## 2026-10-07 — Phase 5

- User confirmed the practice collector and asked to proceed.
- Added read-only DatasetReport/row checks, dataset scanner, duplicate/conflicting
  label checks, provenance summaries, class counts/imbalance warnings and CLI.
- Reused CSV schema/feature-key helpers. Kept numerical preprocessing unchanged.
  Added configurable advisory minimum 200 and imbalance ratio 0.5. No new dependencies.
- Report distinguishes integrity from basic training prerequisites. Non-UNVERIFIED
  source notes are not proof of FSL verification; sample count is not an accuracy claim.
- Ran checker against user's practice.csv: 5 unique HELLO samples, no invalid or
  duplicate rows, 1 participant/session, all 5 marked UNVERIFIED. Integrity PASS;
  basic training checks NOT MET. No actual dataset was changed.
- All 36 tests passed (12 new validation tests). Checks include metadata/length
  mismatches, invalid numerical values, unknown labels, duplicates, imbalance,
  provenance/session issues, CLI statuses and byte-for-byte read-only behavior.
  compileall and CLI --help also passed. No classifier or training implemented.
- Remaining: verified multi-class collection across sessions and validation review
  before real training. Random Forest training is Phase 6.

## 2026-10-07 — Phase 6

- User explicitly requested Phase 6 training implementation.
- Added Random Forest fitting with shared dataset validation, Pandas feature
  selection (f1..f126 only), held-out metrics and separate evaluator module.
- Explained a session-stratified default split to prevent recordings from spanning
  both sets. Preserved an explicit stratified 80/20 frame baseline with overlap
  warnings. Recorded actual fractions when whole sessions require a different size.
- Saved fitted model with joblib, feature/preprocessing metadata, dataset/model
  hashes, split record IDs, library versions, class counts and overlap information.
  Added classification report text/JSON and confusion-matrix PNG. Existing runs
  are preserved; unverified experiments require practice mode and separate outputs.
- Installed Pandas 3.0.6, scikit-learn 1.9.1, joblib 1.6.0 and dependencies into
  the existing virtual environment; explicitly pinned existing matplotlib 3.11.2.
- First test attempt occurred before pip had completed and failed on missing
  joblib. Waited for installation completion and reran: all 46 tests passed,
  including ten new split/training/evaluation/persistence/CLI checks. pip check,
  compileall and CLI help passed. No architecture rewrite was needed.
- Actual practice.csv (five HELLO rows) was correctly rejected as one-class input;
  no real model or accuracy was produced and the user's dataset was unchanged.
- Ran a complete 200-tree synthetic software smoke test under workspace work/,
  with 40 arbitrary examples, 32 train and 8 test, disjoint recording sessions.
  Inspected the confusion-matrix PNG: labels and counts readable, practice scope
  stated in its title. These synthetic patterns provide no FSL evidence.
- Remaining: collect verified multi-class examples across sessions, validate and
  run actual training. Dedicated evaluation and runtime loading/inference are later
  phases; no webcam sign classifier was added in this phase.

## 2026-10-07 — external alphabet dataset inspection

- User downloaded/extracted Kaggle japorton/fsl-dataset and requested inspection.
- Found Collated/A..Z, 450 JPEGs per class (11,700 total). All images decoded.
  Actual dimensions vary; found 555 extra byte-identical files and 1,063
  multi-file name-derived source groups. Original files were not modified.
- No licence or signer/session documentation files were bundled. Sample inspection
  showed image-quality variation and mirrored/brightness variants. Six selected
  original-file inference checks yielded one usable hand result; this is not a
  dataset-wide success rate. Thresholds and core preprocessing remained unchanged.
- Recorded findings in KAGGLE_DATASET_INSPECTION.md. Next: image-mode importer,
  separate alphabet vocabulary, extraction/duplicate audit and explicit source-group
  evaluation support. No sessions/signers will be invented from individual filenames.

## Alphabet image import adapter — 2026-10-07
Implemented independent IMAGE-mode detection, separate alphabet schema, auditable filtering and source grouping, and automatic group splits for practice training. The 54-test suite passes, including the original webcam and training tests. Raw images remain unchanged. Full import and dataset integrity validation are recorded in the generated reports.

## Live alphabet model integration — 2026-10-07
Added run_live.py and LivePredictor: model/schema/checksum validation, shared webcam preprocessing, score filtering, majority smoothing and cleanup. Default practice alphabet model loads successfully. All 59 tests pass. Hardware webcam display remains for user verification.

## Webcam alphabet adaptation — 2026-10-07
Added alphabet vocabulary mode to the collector with a separate webcam schema/dataset, schema-aware validation and live-model loading. Session holdout remains automatic. Added collection-to-training-to-runtime integration coverage. Real webcam recording is the next user step.

## Timed targeted collection and combined training — 2026-10-07
Added S-controlled timed capture with countdown, pauses and target stop. Added adaptation training preserving imported classes and holding out webcam recordings, test overlap filtering and baseline comparison. Original baseline and data preserved.

## Single-hand experiment — 2026-10-08
Audited hand slots; discovered O webcam samples overwhelmingly have two detected hands. Added explicit experimental right-to-left x reflection, versioned runtime transform and separate saved model. Existing v2 preserved. Added visible live hand count. Comparison and limitations documented in SINGLE_HAND_EXPERIMENT.md.

## Phase 11 — Message construction — 2026-10-08
Added explicit stable-letter confirmation and in-memory 200-character message buffer, spacing/delete/clear controls and a footer beneath the camera. Default live model updated to user-tested v3; other model paths remain selectable. All 67 tests pass. Hardware keyboard/display verification remains for the user. Text-to-speech remains Phase 12.

## Phase 12 — Local text-to-speech — 2026-10-08
Added T/Esc controls, hidden asynchronous Windows System.Speech helper, status display and --no-speech. No added packages or external service. Checked installed Microsoft Zira voice and RemoteSigned policy; actual playback awaits user verification. All 71 tests pass.

## Phase 13 — Desktop interface — 2026-10-08
Added Tkinter camera/message interface, action buttons, confidence bar, hand count, keyboard shortcuts and explicit unavailable-camera state. Existing v3 model, recognition and speech modules reused. No additional dependencies. All 74 tests pass, including hidden-window rendering and controls. Visible webcam/layout verification remains for the user.

## Sinabtanay branding and interface redesign — 2026-10-08
Renamed the desktop app Sinabtanay and redesigned visual hierarchy, palette, typography and control grouping. Updated preview titles. Existing behavior and model preserved. Full regression checks and geometry checks cover rendering and control layout.
