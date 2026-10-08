# Extracted Kaggle FSL dataset — inspection, 2026-10-07

Location: data/raw/kaggle_fsl/Collated/. Original source files were not changed.

## Verified from local files

- 26 folders named A through Z, with exactly 450 JPEG files per folder.
- 11,700 files total; all decoded successfully with OpenCV.
- Image dimensions vary. Examples include 300x300, 224x224, and smaller crops;
  the importer must use each image's actual width and height.
- SHA-256 comparison found 555 extra byte-identical files across the archive
  (beyond one copy of each distinct byte sequence). Deduplication needs label
  conflict checks and an audit trail, not blind deletion.
- No README, licence, CSV, JSON, or other documentation files in the extracted
  contents. Download-page licence/provenance still needs separate verification.
- Grouping by (letter folder, filename prefix before _jpg.rf.) produced 9,644
  tentative filename groups; 1,063 groups contain multiple filenames. Among those,
  73 groups contain four files, 847 contain three, and 143 contain two.
  These are name-derived relationships, not known recording sessions or signers.

## Visual/sample checks

Inspected three A/100_jpg.rf.* images and selected plain-named A, B, C, Y, Z
images. Samples include blur, rotated views, and mirrored/brightness-varied
images sharing a prefix. Visual inspection did not verify linguistic labels.

Spot-tested A/100.jpg, B/166.jpg, C/298.jpg, D/460 - Copy.jpg, E/607.jpg, and
Y/5052.jpg using the installed MediaPipe model in IMAGE mode, thresholds 0.6,
current mirror setting and shared preprocessing. B/166.jpg returned one hand
and 126 usable features; the other five returned no hands. This selected sample
is not a random dataset-wide detection-rate estimate. No thresholds were changed.

## Implications for the importer

1. Use a separate alphabet vocabulary/dataset; preserve earlier word practice data.
2. Start with the static baseline's 24 letters excluding J/Z; do not claim dynamic
   letter recognition from individual images. This leaves 10,800 source files
   before quality filtering and deduplication.
3. Use IMAGE mode for independent stills; do not carry VIDEO tracking state between
   unrelated files. Reuse the existing feature normalization and report failed
   detection/preprocessing by class.
4. Track filename and content hashes. Keep identical/related versions on the same
   side of evaluation and check contradictory labels before deduplication.
5. Do not invent participant/session IDs. Name-derived source groups reduce obvious
   augmentation leakage but do not establish session/signer separation. The
   current session-based trainer needs an explicit import schema/split extension.
6. Review retained class counts and later test on separately collected webcam data.

No importer, vocabulary change, training CSV or new model was created by this
inspection. Intermediate preview images were saved only under workspace work/.
