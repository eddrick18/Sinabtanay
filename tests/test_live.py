import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from pathlib import Path
import json
import numpy as np
from config import settings
from src.recognition.predictor import LivePredictor
from scripts.run_live import run_live

ROOT = settings.PROJECT_ROOT / 'models' / 'alphabet_baseline'

@unittest.skipUnless((ROOT / "trained/sign_classifier.joblib").is_file() and settings.ALPHABET_DATASET_PATH.is_file(), "Local baseline model and dataset required for live integration checks.")
class LiveTests(unittest.TestCase):
    def test_real_model_and_smoothing(self):
        p=LivePredictor(ROOT)
        row=np.loadtxt(settings.ALPHABET_DATASET_PATH,delimiter=',',skiprows=1,usecols=range(11,137),max_rows=1)
        p.threshold=0.001
        for _ in range(4):
            self.assertIsNone(p.predict(row)[0])
        self.assertIn(p.predict(row)[0],p.classes)
        p.reset()
        self.assertIsNone(p.predict(row)[0])
        with self.assertRaises(ValueError): p.predict(np.zeros(126))
        self.assertEqual(len(p.history),0)

    def test_low_confidence_and_changed_label_clear_display(self):
        p=LivePredictor(ROOT,window=3,votes=2)
        p.model=MagicMock()
        scores=np.zeros(24); scores[0]=0.9
        p.model.predict_proba.return_value=[scores]
        x=np.ones(126)
        p.predict(x)
        self.assertEqual(p.predict(x)[0],p.classes[0])
        scores=np.zeros(24); scores[1]=0.9
        p.model.predict_proba.return_value=[scores]
        self.assertIsNone(p.predict(x)[0])
        p.model.predict_proba.return_value=[np.full(24,1/24)]
        self.assertIsNone(p.predict(x)[0])
        self.assertEqual(len(p.history),0)

    def test_incompatible_metadata_refused_before_loading(self):
        metadata=json.loads((ROOT/'metadata/model_metadata.json').read_text())
        metadata['schema']['mirror_camera']=False
        with patch.object(Path,'read_text',return_value=json.dumps(metadata)), patch('src.recognition.predictor.joblib.load') as load:
            with self.assertRaises(ValueError): LivePredictor(ROOT)
            load.assert_not_called()

    def test_no_hand_loop_and_cleanup(self):
        detector=MagicMock(); detector.detect.return_value=SimpleNamespace(hand_landmarks=[])
        camera=MagicMock(); camera.read.return_value=(True,np.zeros((480,640,3),dtype=np.uint8))
        with patch('scripts.run_live.HandDetector',return_value=detector), patch('scripts.run_live.cv2.VideoCapture',return_value=camera), patch('scripts.run_live.draw_hand_results'), patch('scripts.run_live.cv2.imshow'), patch('scripts.run_live.cv2.waitKey',return_value=ord('q')), patch('scripts.run_live.cv2.destroyAllWindows') as close:
            self.assertEqual(run_live(ROOT,0,.7,7,5),0)
            camera.release.assert_called_once(); detector.close.assert_called_once(); close.assert_called_once()

    def test_camera_failure_cleanup(self):
        detector=MagicMock(); camera=MagicMock(); camera.isOpened.return_value=False
        with patch('scripts.run_live.HandDetector',return_value=detector), patch('scripts.run_live.cv2.VideoCapture',return_value=camera), patch('scripts.run_live.cv2.destroyAllWindows'):
            self.assertEqual(run_live(ROOT,0,.7,7,5),1)
            camera.release.assert_called_once(); detector.close.assert_called_once()

if __name__ == '__main__': unittest.main()
