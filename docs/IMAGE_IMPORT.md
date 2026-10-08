# Importing the extracted alphabet images

The image importer converts the extracted Kaggle JPEGs into numerical landmarks. It preserves the raw files and writes a separate alphabet dataset; the word-label dataset is unchanged.

Run from the project folder:

```powershell
.\.venv\Scripts\python.exe scripts\import_images.py
.\.venv\Scripts\python.exe scripts\check_dataset.py --dataset data\processed\alphabet_landmarks.csv
```

The importer refuses to overwrite existing outputs. Use `--source` and `--output` for a different source or destination.

## Outputs

- `data/processed/alphabet_landmarks.csv`: accepted examples with 126 normalized features, labels, image dimensions and source identifiers.
- `alphabet_landmarks.metadata.json`: preprocessing contract, source URL, detection settings, model hash and import summary.
- `alphabet_landmarks.audit.csv`: one row per source image, including its disposition and reason.
- `alphabet_landmarks.import_report.json`: totals and per-class counts.

J and Z are excluded because this static-image baseline does not model their movement. No-hand detections, invalid landmarks, duplicates and conflicting labels are withheld. Detection thresholds stay at the project's existing settings. Imported images use independent MediaPipe IMAGE calls, the same mirroring setting and the same wrist/scale/aspect normalization as the webcam pipeline.

Participant, recording session and capture time remain UNKNOWN. Import time is recorded separately. The download did not include a licence file, and its labels have not been independently verified against an FSL reference. This is a practice alphabet baseline, not verified word recognition or full FSL translation.

## Related images and evaluation

Filename families and identical file contents are connected into source groups. Identical rounded feature vectors also connect their source families; conflicting groups are withheld. Training's automatic split uses these groups for imported data, keeping recognized variants together. These groups are heuristic: they cannot establish unseen-signer or unseen-session performance, and unrecognized related images can still inflate evaluation scores. A later live-camera evaluation is necessary.

After inspecting the validation report, the next training command is:

```powershell
.\.venv\Scripts\python.exe scripts\train_model.py --dataset data\processed\alphabet_landmarks.csv --practice --output-dir models\alphabet_baseline
```

The practice flag acknowledges unverified provenance. It does not verify the signs. Training selects the alphabet feature contract and group split automatically. Runtime prediction is a later step.

## Code guide

`src/dataset/image_importer.py` inventories and hashes inputs, creates groups, detects hands, filters candidates, and stages the four outputs. `scripts/import_images.py` provides the command-line entry point. `dataset_utils.py` defines the separate image schema. `validator.py` checks that schema and provenance. `trainer.py` supports group-based splitting while preserving the original webcam session split.

The fixture tests exercise invalid inputs, duplicates, conflicting labels, transitive groups, preserved unknown identities, output protection, validation and practice training.

## Completed import
11,700 source files produced 9,221 valid unique examples across 24 classes and 7,402 source groups. Skipped: 900 J/Z images, 438 duplicate images, 1,138 no-hand images and 3 invalid landmark results. All classes have 286–445 accepted examples. Integrity passes. A seed-42 group split gives 7,376 training rows and 1,845 test rows with every class in both sets and zero source-group overlap. No model was fitted to this dataset during import. See ALPHABET_DATASET_REPORT.txt for validation details.
