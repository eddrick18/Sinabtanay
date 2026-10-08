"""Load a local trained model and filter live predictions."""
import hashlib
import json
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import sklearn
from src.recognition.hand_canonicalization import VERSION, canonicalize
from src.dataset.dataset_utils import current_schema, image_dataset_schema, alphabet_webcam_schema, FEATURE_COLUMNS


class LivePredictor:
    def __init__(self, model_dir: Path, threshold=0.70, window=7, votes=5):
        if not 0 < threshold <= 1 or not 1 <= votes <= window or votes <= window // 2:
            raise ValueError('Use a threshold in (0, 1] and a strict majority of votes within the window.')
        model_path = model_dir / 'trained' / 'sign_classifier.joblib'
        self.metadata = json.loads((model_dir / 'metadata' / 'model_metadata.json').read_text(encoding='utf-8'))
        schema = self.metadata.get('schema')
        self.transform = self.metadata.get('runtime_transform')
        if self.transform not in (None, VERSION):
            raise ValueError('Unknown runtime feature transform.')
        if schema not in (current_schema(), image_dataset_schema(), alphabet_webcam_schema()):
            raise ValueError('Model preprocessing schema is incompatible with this webcam pipeline.')
        if self.metadata.get('sklearn_version') != sklearn.__version__:
            raise ValueError('Model scikit-learn version differs from the installed version. Retrain locally.')
        payload = model_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != self.metadata.get('model_sha256'):
            raise ValueError('Model checksum differs from its metadata.')
        # Load only our locally trained artifact; joblib files can execute code.
        self.model = joblib.load(model_path)
        self.classes = list(map(str, self.model.classes_))
        allowed = schema.get('vocabulary', self.classes)
        if self.model.n_features_in_ != len(FEATURE_COLUMNS) or not set(self.classes) <= set(allowed):
            raise ValueError('Model features or classes are incompatible.')
        self.threshold, self.votes = threshold, votes
        self.history = deque(maxlen=window)

    def reset(self):
        self.history.clear()

    def predict(self, features):
        vector = np.asarray(features, dtype=float)
        if vector.shape != (len(FEATURE_COLUMNS),) or not np.isfinite(vector).all() or not vector.any():
            self.reset()
            raise ValueError('Expected 126 finite, nonzero hand features.')
        if self.transform == VERSION:
            vector = canonicalize(vector)
        probabilities = self.model.predict_proba(vector.reshape(1, -1))[0]
        index = int(np.argmax(probabilities))
        label, confidence = self.classes[index], float(probabilities[index])
        if confidence < self.threshold:
            self.reset()
            return None, confidence, 'Uncertain - hold a clear sign'
        self.history.append(label)
        # A changed current label is never masked by an old stable label.
        if sum(item == label for item in self.history) >= self.votes:
            return label, confidence, 'Stable prediction'
        return None, confidence, 'Hold steady'
