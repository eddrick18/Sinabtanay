# Sinabtanay

A local Windows desktop prototype for real-time recognition of **24 static sign alphabet handshapes**, with fingerspelled message composition and optional local text-to-speech.

Built with Python, OpenCV, MediaPipe, scikit-learn and Tkinter. The model uses 126 normalized hand-landmark features and a Random Forest classifier. Experimental single-hand normalization reflects right-hand features into a consistent slot; its suitability still requires linguistic and live-camera validation.

This is a practice prototype, **not a full Filipino Sign Language translator**. J and Z, dynamic signs, facial expressions and grammar are unsupported. Live model scores are confidence outputs, not measured recognition accuracy. Generalization to unseen signers has not been established.

## Features

- Modern Sinabtanay desktop interface with camera, recognized letter, hand count and message controls.
- Confidence filtering and temporal prediction smoothing.
- Explicit letter confirmation, spaces, deletion and clearing.
- Optional Windows text-to-speech that keeps the webcam responsive.
- Manual or timed webcam landmark collection with countdown and pause.
- Dataset validation, duplicate checks and auditable image import.
- Baseline and combined-data training with grouped or recording-session holdout.
- Automated unit and integration checks.

## Requirements

Windows, a webcam and Python 3.14. The local development environment uses Python 3.14.6; pinned package versions are in requirements.txt. Windows System.Speech supplies optional audio output. No cloud inference service is used.

From your clone's project directory:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\download_hand_model.py
.\.venv\Scripts\python.exe scripts\test_hands.py
```

Only one OpenCV package should be installed: opencv-contrib-python, as pinned. Do not install another cv2 variant into the same environment.

## Models and datasets are separate

This repository contains source code and documentation. It does **not** include the Kaggle images, collected webcam landmarks, generated provenance files, trained joblib models, or the virtual environment. A fresh clone can run camera/hand detection after downloading the MediaPipe hand model, but alphabet prediction requires a trained classifier.

Dataset source: [Kaggle FSL dataset](https://www.kaggle.com/datasets/japorton/fsl-dataset). Review the source's current licence and terms before downloading or redistributing it. The extracted dataset did not include a licence file or signer/session identifiers. Dataset-provided letter labels have not been independently verified as FSL by this project.

Extract the alphabet directories into data/raw/kaggle_fsl/Collated, then:

```powershell
.\.venv\Scripts\python.exe scripts\import_images.py
.\.venv\Scripts\python.exe scripts\check_dataset.py --dataset data\processed\alphabet_landmarks.csv
.\.venv\Scripts\python.exe scripts\train_model.py --dataset data\processed\alphabet_landmarks.csv --practice --output-dir models\alphabet_baseline
.\.venv\Scripts\python.exe scripts\run_app.py --model-dir models\alphabet_baseline
```

Import and training outputs refuse overwrites. Select a different output directory for a new run. The local hand model and the Random Forest classifier are different artifacts; downloading the MediaPipe model alone does not provide sign-letter classification.

## Run the existing local application

If the locally trained v3 model is present:

```powershell
.\.venv\Scripts\python.exe scripts\run_app.py
```

The default is models/alphabet_single_hand_v3. Other compatible runs can be selected with --model-dir. Use --camera-index 1 for another camera, --no-speech to disable audio, or --debug for diagnostic traces. The original OpenCV preview is available through scripts/run_live.py.

| Control | Action |
| --- | --- |
| Enter / Add letter | Confirm the current stable prediction |
| Space | Insert a word space |
| Backspace / Delete | Remove the last character |
| C / Clear | Clear the message |
| T / Speak | Speak the current message |
| Esc / Stop | Stop speech |
| Q / window close | Exit and release resources |

Messages have a 200-character limit and are discarded on exit. Camera images and messages are not saved by the recognition interface. The collector saves numerical landmarks only when collection is explicitly started.

## Targeted webcam adaptation

```powershell
.\.venv\Scripts\python.exe scripts\collect_data.py --vocabulary alphabet --label R --target 30 --timed --interval 1
```

S starts capture after a countdown or pauses it; Q ends a recording. Keep the unused hand out of view and maintain the correct handshape. Record at least two genuinely separate sessions per included letter. After collecting examples:

```powershell
.\.venv\Scripts\python.exe scripts\train_adapted.py --canonical-single-hand --output-dir models\alphabet_experiment
.\.venv\Scripts\python.exe scripts\run_app.py --model-dir models\alphabet_experiment
```

Adaptation keeps the imported alphabet classes and holds out complete webcam recordings for the collected letters. Scores cover that test set only. Repeated tuning against the same recordings needs a new final evaluation session. Do not load joblib models from untrusted sources.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The full local suite has 74 tests. Checks needing excluded datasets or trained artifacts are skipped in a source-only clone; fixture-based tests still run. Desktop integration checks require a Windows desktop session. Synthetic fixtures verify software behavior, not FSL recognition accuracy.

## Project structure

- config/settings.py — camera, feature and training defaults.
- scripts/ — setup, collection, validation, training and application entry points.
- src/hand_tracking/ — MediaPipe detection and shared landmark normalization.
- src/dataset/ — schemas, storage, validation, timed capture and image import.
- src/training/ — classifier training, evaluation and combined-data adaptation.
- src/recognition/ — model loading, filtering, single-hand transform and message buffer.
- src/speech/ — optional asynchronous Windows speech.
- src/ui/ — Sinabtanay desktop interface.
- tests/ — fixture and local integration checks.
- docs/ — implementation walkthroughs, limitations and development history.

Start with [desktop usage](docs/DESKTOP_UI.md), [collection and adaptation](docs/WEBCAM_IMPROVEMENT.md), [image import](docs/IMAGE_IMPORT.md), and [scope and limitations](docs/FSL_LIMITATIONS.md).

## Development context

Developed by Eddrick Miano with AI-assisted implementation and debugging. This repository documents an evolving prototype; reported development-set results must not be presented as full-alphabet, unseen-signer or real-world accuracy guarantees.
