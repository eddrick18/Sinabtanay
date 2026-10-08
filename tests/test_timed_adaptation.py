import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from config import settings
from src.dataset.capture_timer import CaptureTimer
from src.dataset.dataset_utils import image_dataset_schema,alphabet_webcam_schema
from src.training.adaptation import adapt_model
from src.recognition.predictor import LivePredictor

class TimedAdaptationTests(unittest.TestCase):
    def test_timer_countdown_pause_resume_and_no_catchup(self):
        t=CaptureTimer()
        self.assertFalse(t.due(100))
        t.toggle(100)
        self.assertFalse(t.due(102.9))
        self.assertTrue(t.due(103))
        self.assertFalse(t.due(103.5))
        self.assertTrue(t.due(110))
        self.assertFalse(t.due(110))
        t.toggle(110)
        self.assertFalse(t.due(120))
        t.toggle(120)
        self.assertFalse(t.due(122))
        self.assertTrue(t.due(123))
        t.pause()
        self.assertFalse(t.due(130))
        for interval in [.1,float('nan')]:
            with self.assertRaises(ValueError): CaptureTimer(interval)

    @unittest.skipUnless(settings.ALPHABET_DATASET_PATH.is_file() and (settings.PROJECT_ROOT / "models/alphabet_baseline/trained/sign_classifier.joblib").is_file(), "Local dataset and baseline model required.")
    def test_combination_holdout_and_reload(self):
        external=pd.read_csv(settings.ALPHABET_DATASET_PATH,dtype=str,keep_default_na=False).groupby('label').head(5).copy()
        local=external[external.label=='A'].iloc[:4].copy()
        local['session_id']=['recording1','recording1','recording2','recording2']
        local['participant_id']='test_signer'
        external_info={'schema':image_dataset_schema(),'dataset_sha256':'external'}
        local_info={'schema':alphabet_webcam_schema(),'dataset_sha256':'local'}
        baseline=settings.PROJECT_ROOT/'models/alphabet_baseline'
        with tempfile.TemporaryDirectory() as d, patch('src.training.adaptation.load_training_data',side_effect=[(external,external_info),(local,local_info)]), patch.object(settings,'RANDOM_FOREST_TREES',8):
            output=Path(d)/'model'
            m=adapt_model(Path('external.csv'),Path('local.csv'),output,baseline)
            self.assertEqual(m['testing_samples'],2)
            self.assertGreaterEqual(m['excluded_training_rows_matching_test'],2)
            self.assertEqual(m['session_overlap'],0)
            self.assertEqual(len(LivePredictor(output).classes),24)
        local['session_id']='single_recording'
        with patch('src.training.adaptation.load_training_data',side_effect=[(external,external_info),(local,local_info)]):
            with self.assertRaisesRegex(ValueError,'two separate'): adapt_model(Path('e'),Path('w'),Path('o'),baseline)

if __name__=='__main__': unittest.main()
