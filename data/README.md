# Local landmark data

The collector creates processed/landmarks.csv and its .metadata.json schema file.
CSV contains labels, pseudonymous participant/session identifiers, UTC timestamps,
hand count, image dimensions, reference notes, and f1..f126 numerical features.
It contains no webcam image/video. Raw recording is not implemented.

Do not treat pseudonymous landmark data as automatically anonymous. Obtain consent
before collecting from others, agree on retention/sharing, and avoid names or
private details in reference notes. Keep identifiable consent records outside
this project. CSV files are ignored by Git; check any files you intentionally share.

Use a separate practice dataset for unverified gestures. Never merge it into the
verified FSL dataset merely because the column names match. The metadata file
records the preprocessing schema, not proof that a sign was verified.

See ../docs/DATA_COLLECTION.md for capture commands and collection guidance.

External alphabet imports use a separate alphabet_landmarks.csv schema with source file, content hash and heuristic source group. Recording identities and capture times remain UNKNOWN. See ../docs/IMAGE_IMPORT.md.
