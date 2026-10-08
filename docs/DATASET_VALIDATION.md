# Phase 5 — inspect the dataset before training

The checker reads the CSV and its schema metadata. It does not modify samples,
remove duplicates, fill missing values, relabel rows, or train a model.

## Windows commands

From the project folder:

```powershell
.\.venv\Scripts\python.exe scripts\check_dataset.py --dataset data\processed\practice.csv
```

--dataset selects the existing practice file. Without that option the checker
uses data/processed/landmarks.csv, the default real-data collection path. If the
file is missing it explains how to collect samples. Relative paths are relative
to the terminal's current folder.

For collected real data:

```powershell
.\.venv\Scripts\python.exe scripts\check_dataset.py
.\.venv\Scripts\python.exe scripts\check_dataset.py --require-training-ready
```

The second command checks the same file but returns exit status 2 if its basic
training prerequisites are not met. Status 0 means no integrity errors (and,
when required, prerequisites passed). Status 1 means invalid/unreadable data.
Use `$LASTEXITCODE` in PowerShell to inspect the latest command's exit status.
Normal practice warnings alone do not produce a failure exit by default.

--min-samples changes the advisory count from the default 200 per present class.
--imbalance-ratio sets the smallest/largest count warning threshold (default 0.5).
These are configurable engineering checks, not guarantees of accuracy or quality.
--debug includes diagnostic traceback details for unexpected read errors.

## Report interpretation

Integrity checks include:

- Correct CSV columns, 126 feature columns, and identical row lengths.
- Required nonempty values; numerical, finite features; known labels.
- Wrist at origin, unit maximum radius for populated hand slots, nonzero input,
  and consistency between hand_count and populated slots.
- Positive integer frame dimensions, valid UTC timestamps, and consistent
  participant identity for a given recording session.
- Compatible preprocessing version, mirror convention and sidecar schema.
- Exact repeated CSV rows, repeated rounded feature vectors, and conflicting
  labels assigned to the same vector.

Counts per configured label show valid unique examples. Total CSV rows includes
invalid and duplicate records; invalid rows are counted once even if they have
multiple errors. A later duplicate is excluded from valid unique counts. An
integrity failure means the whole file needs review; valid counts are not an
automatically cleaned training dataset. Error details show the first 20 invalid
records, with a count of remaining invalid records. Record numbers are CSV record
ordinals including the header, not necessarily physical lines for multiline fields.

Warnings cover absent labels, small classes, class imbalance, unverified reference
markers, one-participant data, and labels recorded in only one session. Unknown
labels and duplicates are integrity errors. Imbalance is a warning because a
legitimately skewed dataset can still be analyzed; it should be considered before
choosing a training strategy and evaluation metrics.

Basic training checks require no integrity errors, at least two populated classes,
no UNVERIFIED marker among valid samples, the selected minimum count in every
present class, and at least two sessions per present class. Unused configured
labels and single-participant scope remain warnings. The model's eventual
vocabulary is limited to the classes actually present. Passing this check does
not establish FSL correctness, participant diversity, independence between samples,
suitable train/test split, classifier performance, or generalization.

A source note different from UNVERIFIED is only your recorded provenance. The
software cannot authenticate a consultation or verify that a gesture matches it.
Do not change a note just to pass the checker. Validate actual signs and their
suitability for the current representation with FSL references/collaborators.

## Actual practice result — 2026-10-07

The checked practice.csv contains five unique HELLO samples, zero invalid rows,
zero duplicate rows/features, one participant and one session. All five have
UNVERIFIED reference markers. Integrity passes; basic training checks do not.
That is the expected result of the five-sample controls exercise. Preserve this
as practice data rather than merging it into a verified FSL dataset.

## Files and how they connect

| File | Input / purpose | Output / connection |
| --- | --- | --- |
| src/dataset/validator.py | CSV path, settings, schema sidecar, advisory limits | DatasetReport and formatted report; reuses existing schema/feature-key helpers |
| scripts/check_dataset.py | Windows CLI options | Terminal report and exit code; calls validator, handles missing files/read errors |
| tests/test_dataset_validator.py | Temporary CSV fixtures produced with the actual processor | 12 tests of good/bad data, duplicates, provenance, counts, read-only behavior and CLI statuses |
| config/settings.py | MIN_SAMPLES_PER_CLASS and CLASS_IMBALANCE_RATIO | Defaults used by the validator/CLI |
| docs/DATASET_VALIDATION.md | This guide | Explains checks, counts, limitations and the practice outcome |
| docs/PRACTICE_DATASET_REPORT.txt | A snapshot of your actual practice validation | Reviewable terminal-style results without raw landmark data |
| README.md and DEVELOPMENT_LOG.md | Current phase and evidence | Run instructions and recorded engineering work |

DatasetReport groups counters, warnings and errors into one understandable result.
inspect_row checks individual row fields and feature geometry. validate_dataset
reads metadata, scans rows, detects repeats, aggregates counts and adds quality
warnings. format_report converts that result into readable text. The CLI contains
no numerical preprocessing or training logic. Existing dataset writer append
checks remain separate: they protect writing; this checker produces a full review.

The standard-library CSV reader and NumPy suffice at this size. No new package is
needed. Pandas can be introduced when later analysis benefits from data frames.

## What to do next

For real training, collect correctly labelled, verified signs that fit the static
representation, from multiple sessions and ideally multiple consenting signers.
Aim for about 200–500 useful samples per class initially, while prioritizing
variation and label correctness over count. Review errors manually; do not
silently repair data or create fake examples to satisfy thresholds.

Recordings from the same session can be very similar. Session/participant context
must inform a later held-out evaluation plan, rather than become model features.
Keep f1..f126 as X and label as y. Phase 6 will introduce Random Forest training
after this validation stage; no model or accuracy is produced here.
