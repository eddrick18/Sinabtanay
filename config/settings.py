"""Central defaults for webcam and hand detection."""

from pathlib import Path

CAMERA_INDEX = 0
WINDOW_TITLE = "Sinabtanay - Webcam Test"
DEBUG = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HAND_WINDOW_TITLE = "Sinabtanay - Hand Detection"
MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.6
MIN_HAND_PRESENCE_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6
MIRROR_CAMERA = True
PREPROCESSING_VERSION = "wrist-max-radius-aspect-v1"
FEATURE_COUNT = 126  # Left 21x3, then Right 21x3; independent of MAX_HANDS.
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "landmarks.csv"
COLLECTION_WINDOW_TITLE = "Sinabtanay - Data Collection"
COLLECTION_TARGET = 300  # Per recording session; sample count alone does not imply quality.
CAPTURE_COOLDOWN_SECONDS = 0.5
MIN_SAMPLES_PER_CLASS = 200  # Advisory collection goal; not an accuracy guarantee.
CLASS_IMBALANCE_RATIO = 0.5  # Warn if the smallest present class is below half the largest.
RANDOM_STATE = 42
TEST_SIZE = 0.2
RANDOM_FOREST_TREES = 200
RANDOM_FOREST_MIN_SAMPLES_LEAF = 2
MODEL_PATH = PROJECT_ROOT / "models" / "trained" / "sign_classifier.joblib"
MODEL_METADATA_PATH = PROJECT_ROOT / "models" / "metadata" / "model_metadata.json"
# Placeholder labels only: verify actual signs using FSL references/collaborators.
SIGN_LABELS = ["HELLO", "THANK_YOU", "YES", "NO", "HELP", "WATER"]
# Dataset-provided alphabet labels; independent linguistic verification is pending.
ALPHABET_LABELS = [letter for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if letter not in "JZ"]
ALPHABET_SOURCE_PATH = PROJECT_ROOT / "data" / "raw" / "kaggle_fsl" / "Collated"
ALPHABET_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "alphabet_landmarks.csv"
ALPHABET_SOURCE_URL = "https://www.kaggle.com/datasets/japorton/fsl-dataset"
HAND_MODEL_PATH = PROJECT_ROOT / "models" / "pretrained" / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
