# Improving recognition with webcam examples

## First collection trial

From the project folder, close the live preview and run:

```powershell
.\.venv\Scripts\python.exe scripts\collect_data.py --vocabulary alphabet --label A --target 50
```

Focus the preview. Make the correct A handshape, tap Space to save a sample, release Space, slightly vary its position and then tap again. The status must say Saved sample. Keep the letter correct while varying position, distance and small natural angles. Fully include the hand and fingers. Avoid simply holding Space. Q exits. Only numerical landmarks are saved.

This uses data/processed/alphabet_webcam.csv and its schema sidecar. Do not append to the imported alphabet_landmarks.csv. Signer defaults to signer_01: use the same pseudonym for yourself throughout. Samples remain UNVERIFIED unless a real reference is supplied; a label typed by the collector does not establish sign correctness.

After the first 50 A samples, repeat for B and C by changing --label. This is a pilot for checking collection, not a replacement for the full alphabet model. Include any letters that were being confused; record the intended letter, not the model's wrong prediction.

## Collect separate recordings

Make at least three recordings per included letter: finish with Q, reposition and take a break, then launch the command again. Every launch records a new session ID. Do not treat rapid restarts of nearly identical poses as independent conditions. Repeat in different realistic lighting/backgrounds, maintaining clear visibility and correct handshape. Do not add changed or malformed signs under the same label.

Start with 50 varied examples per recording. Counts alone do not guarantee coverage; additional recordings are more useful than many near-identical frames. For a full 24-letter replacement, collect every letter except J/Z in the same dataset. An A/B/C-only model cannot recognize the remaining letters and may assign them one of its known labels.

## Validate, train and compare

Once there are multiple letters and at least three genuine recordings for each:

```powershell
.\.venv\Scripts\python.exe scripts\check_dataset.py --dataset data\processed\alphabet_webcam.csv
.\.venv\Scripts\python.exe scripts\train_model.py --dataset data\processed\alphabet_webcam.csv --practice --output-dir models\alphabet_webcam_v1
.\.venv\Scripts\python.exe scripts\run_live.py --model-dir models\alphabet_webcam_v1
```

Automatic splitting holds out complete per-letter recordings. Practice mode permits the small, unverified experiment; validation can report unmet advisory sample goals. This evaluates other recordings of the same signer, not unseen signers. Keep your original alphabet_baseline model for comparison; it is never overwritten.

Compare both models with fresh webcam attempts: note intended letter, displayed letter, uncertain/blank results, hand used and lighting. Do not directly compare their dataset accuracy percentages because the test sets differ. If you repeatedly tune on the held-out recording, collect another fresh evaluation recording before making final accuracy claims.

## Implementation

collect_data.py accepts --vocabulary words or alphabet and selects a separate default dataset. DatasetWriter binds labels to the selected schema and refuses mixing schemas. Alphabet webcam schema preserves actual recording identity, timestamp and dimensions, while the imported image schema retains UNKNOWN recording context. The validator, trainer and LivePredictor support both. The integration fixture records numerical examples, validates them, trains with session holdout and reloads the new artifact; it does not create real user training data.

## Faster targeted collection and combined adaptation

You no longer need to tap Space for every example:

```powershell
.\.venv\Scripts\python.exe scripts\collect_data.py --vocabulary alphabet --label A --target 50 --timed
```

Click the preview, form the correct letter, and press S. Capture begins after three seconds and attempts one sample every 0.75 seconds. S pauses or resumes (with a fresh countdown); Q exits. Target completion pauses automatically. No-hand/invalid/duplicate attempts do not count toward the target. Vary position, distance and small natural angles while preserving the correct sign. Pause before changing your handshape. Timed sampling improves convenience, not independence or data quality. Options --interval and --countdown adjust timing; interval must be at least 0.5 seconds.

Your earlier 50 A samples count as the first recording. Record another A session using the timed command. Prioritize letters that fail and their confusing alternatives; change --label for each. For every included letter, collect at least two genuinely separate recordings, ideally three or more, with a break and repositioning. The model never guesses labels for training examples: your selected label is applied to each accepted capture.

Once those recordings exist, combine them with the imported dataset:

```powershell
.\.venv\Scripts\python.exe scripts\train_adapted.py
.\.venv\Scripts\python.exe scripts\run_live.py --model-dir models\alphabet_adapted_v1
```

This preserves all 24 imported letter classes and adds webcam training examples. A reproducible seed-42 selection holds out one complete webcam recording for EACH collected letter. Matching test feature vectors are removed from training, along with related imported source groups. Exact training duplicates are deduplicated and contradictory training labels cause an error. No test examples are used for fitting. Imported recording relationships remain unknown, so unrecognized related images remain a limitation.

The saved metadata records input hashes, held-out session IDs, test letters, exclusions, and both baseline and adapted metrics on the SAME webcam test set. This comparison covers only collected letters and recordings; it does not prove full-alphabet or unseen-signer performance. A new model is saved separately; the original baseline and collected data are unchanged. If adapted results are worse, continue using the baseline and improve the examples rather than assuming retraining guarantees improvement. Repeated tuning needs a fresh final test recording.

Implementation: capture_timer.py handles nonblocking timing; adaptation.py combines verified-compatible feature schemas, holds out recordings, trains and compares; train_adapted.py is its command-line entry point. No real adaptation run is performed until your required recordings exist.
