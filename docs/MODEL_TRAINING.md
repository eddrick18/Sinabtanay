# Phase 6 — Random Forest baseline training

Pipeline: validated CSV → feature matrix X and labels y → stratified holdout split
→ fit Random Forest on training rows → evaluate test rows → save model and reports.
No webcam inference, smoothing, speech, hyperparameter search or deep-learning
training is implemented here. A later dedicated evaluation/loading phase can build
on the saved split and metadata; basic held-out evaluation belongs to this baseline.

## Current data and what you can run

Your existing practice.csv has five HELLO examples and no second class. That
checks collection controls but cannot train a meaningful classifier: predicting
one label for every input would not discriminate signs. --practice does not
bypass the two-class requirement, integrity errors or split feasibility.

Dependencies were added to the existing environment. On another Windows setup,
install the updated requirements using that environment's interpreter:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For real training, collect correctly labelled verified examples into the default
landmarks.csv with at least two labels and at least two recording sessions per
label. Relaunching collect_data.py creates a separate session. Keep consistent
participant IDs. Aim initially for 200–500 useful examples per class across
sessions, but do not treat count as proof of linguistic correctness or accuracy.
The trainer reports small-count warnings rather than enforcing 200 as a hard
mathematical minimum. It refuses UNVERIFIED references in normal mode.

Run from the project folder in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\check_dataset.py
.\.venv\Scripts\python.exe scripts\train_model.py
```

The checker summarizes the real dataset. The trainer independently revalidates
it, makes a session holdout, trains and prints per-class results. A missing
landmarks.csv gives a clear collection instruction, not a traceback by default.

To see the current practice file's clear rejection:

```powershell
.\.venv\Scripts\python.exe scripts\train_model.py --dataset data\processed\practice.csv --practice
```

For a future multi-class software experiment, use a separate practice CSV and
--practice. Outputs go to models/practice by default and metadata/plot/report
explicitly mark practice. These metrics are not FSL accuracy. Do not relabel
arbitrary hand poses as verified signs or edit a reference merely to bypass checks.

## Splitting and leakage

Default `--split session` uses train_test_split on session IDs, stratified by each
session's label, then maps the chosen sessions back to their rows. Our collector
records one label per session, so the trainer requires that convention. Mixed-label
sessions need a different group-aware strategy; the program explains this rather
than silently falling back to a frame split. Every present class must have at
least two sessions and must appear in both train and test.

The requested test fraction is 20%, but sessions remain whole and at least one
test session per class is required. With two classes recorded twice each,
two of four sessions are held out: approximately 50%, not 20%, if session sizes
match. Five equally sized sessions per class allow the intended 80/20 division.
Unequal session sizes change the actual row fraction. The console and metadata
record the actual fraction; there is no guarantee of exactly 80/20 session rows.

The requested conventional frame baseline remains available explicitly:

```powershell
.\.venv\Scripts\python.exe scripts\train_model.py --split frame --output-dir models\frame_baseline
```

It stratifies rows by label with the selected fraction. Train/test row identities
remain disjoint, and the validator rejects exact duplicates. However, similar
captures from the same session can appear in both sets, making results optimistic.
The trainer warns and records session overlap. Use this for a baseline comparison,
not as evidence of unseen-session or unseen-signer generalization.

Session holdout also does not automatically hold out people. The same participant
may be in both sets; overlap is reported and recorded. A later held-out-signer
experiment is needed for claims about new users. Do not repeatedly adjust the
model based on these test results and continue calling that same data an untouched
test set. Later tuning needs a separate validation set or cross-validation.

## Understanding X, y, fit and predict

Pandas reads the validated CSV. X contains only f1..f126; its shape is
(number_of_samples, 126). y contains the corresponding label for each sample.
Participant, session, timestamp, dimensions and source notes are context, not
predictive inputs. Including them could let the classifier exploit recording
habits rather than sign geometry. Training reads the saved normalized features;
it does not normalize them again or train on webcam images.

Random Forest fits many decision trees on bootstrapped examples and subsets of
features. Their combined predictions reduce reliance on one tree. It is a
reasonable conventional baseline for compact numerical features and does not
require a neural-network training framework. Tree thresholds do not require
StandardScaler here. Existing wrist/scale preprocessing still matters because
it defines the meaning of our landmark representation.

The defaults are 200 trees and a minimum of two samples per leaf. The latter
reduces the tendency to create leaves for single training examples but cannot
guarantee freedom from overfitting. random_state=42 controls random splitting
and forest construction; package versions, dataset hash and parameters are saved
for reproducibility. n_jobs=-1 uses available CPU workers during forest fitting.
--trees and --test-size change the corresponding defaults.

`model.fit(X_train, y_train)` learns the trees using training rows only.
`model.predict(X_test)` returns labels for the held-out examples. Test labels
are used for evaluation, not fitting. The saved model is this training-split
model; we do not refit on all rows after measuring the test score.

Sources: [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
and [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).

## Reading the metrics

- Accuracy: fraction of test samples classified correctly; it can hide weak classes.
- Precision for HELP: of the samples predicted as HELP, how many actually were HELP?
- Recall for HELP: of the actual HELP samples, how many did the model find?
- F1-score: harmonic mean of precision and recall; both need to be good.
- Support: number of actual test examples of that class.

Macro averages weight each class equally; weighted averages weight by support.
A confusion matrix has actual classes on rows and predicted classes on columns.
Diagonal entries are correct predictions; other entries show confusions. Label
order is identical in model metadata, report and matrix. Undefined class metrics
are represented by zero with zero_division=0; inspect support and errors rather
than interpreting a zero metric without its context.

Source: [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html).

## Saved outputs

The normal default locations are:

```text
models/trained/sign_classifier.joblib
models/metadata/model_metadata.json
models/metadata/evaluation.json
models/metadata/classification_report.txt
models/metadata/confusion_matrix.png
```

--output-dir selects another root with the same trained/ and metadata/ subfolders.
If any expected output exists, training refuses to overwrite it. Use a new output
directory for subsequent experiments. Run one trainer per output directory at a
time. Outputs are staged first, then published with metadata last; disk errors
during publication can still leave an incomplete run that must not be used.

joblib saves the fitted model. Metadata records model labels, 126 features and
column order, preprocessing/mirror schema, dataset/model SHA-256, UTC creation
time, library versions, parameters, train/test row records and class counts,
split fractions, overlap counts, warnings and practice status. CSV record IDs
plus dataset hash identify the exact holdout. No raw images are stored.
Future loading must verify feature/schema compatibility, model hash and practice
status before inference. Load only trusted model files you created; persistence
is not a safe interchange format for arbitrary downloaded files.

## File explanations

| File | Input / responsibility | Output / connections |
| --- | --- | --- |
| src/training/__init__.py | Defines training package | Allows focused training/evaluation imports |
| src/training/trainer.py | Validated CSV/options; loads context with Pandas, splits, fits, stages outputs | Saved fitted Random Forest, metadata and reports; reuses validator and evaluator |
| src/training/evaluator.py | Actual test labels, predicted labels, fixed label order | Accuracy, precision/recall/F1/support, confusion counts and PNG; no fitting |
| scripts/train_model.py | Terminal options and settings | Runs trainer, prints results, returns 0 on success or 1 on failure |
| tests/test_trainer.py | Temporary synthetic landmark datasets | Ten engineering tests of splitting, fitting/save/reload, metrics, refusal cases and CLI |
| requirements.txt | Pinned Pandas, scikit-learn, joblib and matplotlib | Dependencies in the existing virtual environment |
| config/settings.py | Seed, fraction, tree/leaf defaults, model paths | Training defaults; output-dir customizes root |
| .gitignore | Generated models/reports | Keeps binary models and dataset-linked run records out of Git |
| docs/MODEL_TRAINING.md / README.md / DEVELOPMENT_LOG.md | Explanations and validation evidence | Repeatable workflow and honest phase status |

load_training_data checks integrity and class/provenance prerequisites, then reads
only a stable snapshot. split_dataset returns row indices. train_model extracts
only feature columns, fits on the training rows and evaluates the holdout.
save_training_run stages model/reports and records the model hash. The evaluator
owns metric calculations and plot rendering so it can later be reused.

## Validation scope

Synthetic test fixtures are arbitrary geometry with placeholder labels. They
exercise the pipeline and model persistence but establish no FSL recognition
performance. No trained real FSL model can be produced from the current one-class
practice file. Real dataset collection and evaluation remain necessary.
