"""Alphabet recording contract and session training integration."""
import tempfile
import unittest
from pathlib import Path
from src.dataset.dataset_utils import DatasetWriter
from src.dataset.collector import SampleCollector
from src.dataset.validator import validate_dataset
from src.training.trainer import train_model
from src.recognition.predictor import LivePredictor
from test_dataset_utils import hand_result

class AlphabetWebcamTests(unittest.TestCase):
    def test_collect_validate_train_and_load(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'alphabet.csv'
            writer=DatasetWriter(path,'alphabet')
            for i,label in enumerate(['A','B']):
                for session in range(2):
                    collector=SampleCollector(writer,label,'test_signer',f'{label}_{session}','UNVERIFIED',3)
                    for n in range(3):
                        collector.capture(hand_result(variation=.01*(1+i*6+session*3+n)),(640,480),n)
            self.assertEqual(DatasetWriter(path,'alphabet').sample_counts,{'A':6,'B':6})
            with self.assertRaises(ValueError): DatasetWriter(path)
            with self.assertRaises(ValueError): SampleCollector(writer,'J','s','r','UNVERIFIED',3)
            report=validate_dataset(path)
            self.assertEqual(report.errors,[])
            self.assertEqual(report.valid_rows,12)
            output=Path(d)/'model'
            train_model(path,output,practice=True,tree_count=8)
            predictor=LivePredictor(output)
            self.assertEqual(predictor.metadata['split_mode'],'session')
            self.assertEqual(set(predictor.classes),{'A','B'})

if __name__=='__main__': unittest.main()

