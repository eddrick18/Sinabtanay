# Phase 4 — controlled dataset collection

The objective is to record deliberate examples of selected, verified isolated
FSL signs. This phase stores landmarks only. It does not train a model or identify
which sign is currently visible: the label comes from your terminal command.
Always perform that label's verified sign when capturing.

## Safe first test on Windows

From Windows PowerShell:

```powershell
Set-Location 'C:\Users\PREDATOR\Documents\Codex\2026-10-07\files-pasted-by-the-user-project\outputs\signbridge-fsl'
.\.venv\Scripts\python.exe scripts\collect_data.py --label HELLO --target 5 --dataset data\processed\practice.csv
```

This uses the installed environment, selects the placeholder HELLO label, sets
a five-sample session target, and writes a separate practice dataset. Without a
--reference argument it records UNVERIFIED. That is useful for checking controls;
it does not establish an FSL gesture. No particular gesture is prescribed here.

Focus the window. With hands out of view tap Space: status should say skipped
and counts should stay unchanged. Show a valid hand, tap Space once, and check
the count rises. Wait at least half a second between taps. Try slightly varied
poses, then Q. You should see a CSV and matching schema metadata next to it.

For verified data, replace the example note with a real source/consultation note:

```powershell
.\.venv\Scripts\python.exe scripts\collect_data.py --label HELLO --participant-id signer_01 --reference "Actual verified source and sign variant" --target 100
```

The quoted reference is a template to replace, not evidence of verification.
Use qualified FSL users, interpreters, teachers, Deaf collaborators, or reliable
FSL references to verify each actual sign and variant. Do not infer a sign from
its English label. The initial labels in settings are placeholders; edit SIGN_LABELS
to your verified vocabulary. Remember that movement and location-dependent signs
may not be suitable for our current static, wrist-relative representation.

--participant-id is a stable pseudonym: use the same ID for the same participant
across sessions, and different IDs for different people. Each launch generates
a new random session ID. --target limits saved examples for the selected label
in that launch (default 300); it is not a minimum accuracy requirement. Label-total
counts include all previous sessions. --dataset chooses another CSV; relative
paths resolve from your terminal's current directory, while the default path
is anchored to the project. --camera-index and --debug work as in the hand preview.

## What happens on Space

1. The OpenCV loop reads and detects a fresh frame, then shows it.
2. A Space key event requests capture of that same frame's result.
3. The collector checks cooldown, target, and presence of hands.
4. The shared create_feature_vector function validates, normalizes and orders landmarks.
5. The writer rejects an identical rounded vector already saved under this or
   another label; contradictory labels get a different explanatory status.
6. The writer appends one CSV row, flushes it, updates counts, and reports success.

There is no automatic frame recording. Tap Space; do not hold it down. OpenCV
reports key events rather than reliable key-up state, so OS key repeat can request
additional captures when held. A 0.5-second cooldown limits repeated capture
requests but does not guarantee one save per physical held press. The default
target stops additional saves but leaves the window open until Q.

Hands must be fully visible. For a two-hand sign, check both hands are detected
before tapping Space: the collector accepts one or two hands but cannot know
how many your sign requires. It also cannot assess whether you performed the
correct sign. Unknown/duplicate handedness or malformed landmarks are skipped.

## Data contract and provenance

The actual columns are:

```text
label,participant_id,session_id,captured_at_utc,hand_count,frame_width,frame_height,source_reference,f1,...,f126
```

Only f1..f126 belong in the future classifier's feature matrix. Labels are the
target y. Context columns are for auditing and grouped evaluation, not predictive
features. Keeping participant/session IDs helps detect leakage: adjacent examples
from one person/session can be strongly correlated. A frame-random test split
can exaggerate generalization, even when it is stratified. Later training should
consider held-out sessions and held-out signers appropriate to the intended use.

The sidecar landmarks.metadata.json records preprocessing version, feature count,
Left/Right ordering, mirror convention, columns, and six-decimal feature storage.
The writer checks metadata/header and basic existing row validity before appending.
If incompatible or damaged, it stops with a clear error. Do not delete metadata
to force incompatible appends; use a new dataset or investigate the file.

Exact duplicate protection compares all 126 rounded values and survives restarts.
It does not eliminate near-duplicates, ensure varied samples, or establish correct
labels. Since normalization removes translation and uniform scale, such changes
alone may legitimately produce the same features and be rejected. Do not add
artificial numerical noise just to increase sample counts.

Run only one collector per dataset at a time; concurrent CSV writers are not
supported. Each saved row is flushed immediately, but a crash/storage failure can
still leave a damaged file; keep backups of useful datasets. Existing malformed
rows are rejected on reopening. Full summary, imbalance, and duplicate reporting
belongs to Phase 5, not this append check.

## Collection quality

Aim initially for about 200–500 useful examples per class across multiple sessions,
not hundreds of consecutive near-identical poses. For example, 10 signs with
300 samples each yield 3,000 examples, but that count alone guarantees nothing.
Use --target to divide collection into manageable sessions and take breaks.

Include modest variations in position, camera distance, orientation, lighting,
background and recording day. Preserve the verified sign's linguistic identity;
vary capture conditions rather than changing it into a different sign. The camera
pipeline can still be affected by lighting/background even though stored inputs
are landmarks. Keep mirror convention consistent and verify handedness.

If multiple users are intended, include multiple consenting participants. A model
trained only on your hands may learn your habits and fail on others. Record sign
variants carefully; consult FSL collaborators about which variations share a
label. Avoid mixing unfinished movements, transitions or wrong labels into clean
isolated-sign examples. Keep practice data separate from verified data.

## File explanations

| File | Input and purpose | Output and connection |
| --- | --- | --- |
| scripts/collect_data.py | Label/options plus webcam; owns capture/display/key loop | Status window, logs and exit status; calls detector and SampleCollector; releases resources |
| src/dataset/__init__.py | No inputs; defines package | Allows dataset module imports |
| src/dataset/collector.py | MediaPipe result, dimensions, capture time, label/session context | Save/skip message; calls shared processor and DatasetWriter only on a capture request |
| src/dataset/dataset_utils.py | Features/context, CSV path and existing schema/rows | Compatible appended numerical rows, schema sidecar and counts; no webcam dependency |
| tests/test_dataset_utils.py | Temporary files and synthetic detections/camera/key events | Nine checks of persistence, no-hand rejection, duplicate protection, cooldown/target, damaged/incompatible files and UI orchestration |
| config/settings.py | Vocabulary placeholders, target, cooldown, paths/window title | Defaults used by CLI, collector and writer |
| data/README.md | Data format and privacy policy | Explains locally stored data and safe sharing |
| docs/DATA_COLLECTION.md | This guide | Commands, file roles and data-quality explanation |
| README.md / DEVELOPMENT_LOG.md | Current phase/evidence | Updated run instructions and engineering history |

In dataset_utils, feature_key validates shape/finite values and rounds numbers for
comparison/storage. current_schema defines the append contract. DatasetWriter loads
compatible rows and counts once; save checks one request and writes it. SampleCollector
enforces capture policy and reuses the Phase 3 processor. The UI draws a separate
header above the image, so its text does not hide the hand skeletons.

## Verification and next phase

Run ` .\.venv\Scripts\python.exe -m unittest discover -s tests -v ` from the project
folder. Tests use temporary data, not your real dataset. The simulated UI test
checks Space-only saving, no-hand skipping and cleanup; it does not demonstrate
physical camera performance. Confirm the manual practice test and inspect saved
counts before Phase 5 dataset validation. No classifier or full dataset report
is implemented in Phase 4.
