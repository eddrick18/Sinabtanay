import unittest
import numpy as np
from config import settings
from src.recognition.hand_canonicalization import canonicalize,VERSION
from src.recognition.predictor import LivePredictor

class CanonicalTests(unittest.TestCase):
    def test_mirror_pair_maps_to_same_features(self):
        hand=np.arange(63,dtype=float).reshape(21,3)/63
        left=np.r_[hand.reshape(-1),np.zeros(63)]
        mirrored=hand.copy(); mirrored[:,0]*=-1
        right=np.r_[np.zeros(63),mirrored.reshape(-1)]
        np.testing.assert_array_equal(canonicalize(left),canonicalize(right))
        np.testing.assert_array_equal(canonicalize(canonicalize(right)),canonicalize(right))

    def test_two_hand_inputs_unchanged_and_invalid_rejected(self):
        both=np.ones(126)
        np.testing.assert_array_equal(canonicalize(both),both)
        with self.assertRaises(ValueError): canonicalize(np.ones(63))
        with self.assertRaises(ValueError): canonicalize(np.full(126,np.nan))

    @unittest.skipUnless((settings.PROJECT_ROOT / "models/alphabet_single_hand_experiment/trained/sign_classifier.joblib").is_file() and settings.ALPHABET_DATASET_PATH.is_file(), "Local experiment artifacts required.")
    def test_real_experiment_loads_and_predicts(self):
        p=LivePredictor(settings.PROJECT_ROOT/'models/alphabet_single_hand_experiment')
        self.assertEqual(p.transform,VERSION)
        vector=np.loadtxt(settings.ALPHABET_DATASET_PATH,delimiter=',',skiprows=1,usecols=range(11,137),max_rows=1)
        label,score,status=p.predict(vector)
        self.assertTrue(0<=score<=1)

if __name__=='__main__': unittest.main()
