"""Train with imported and webcam alphabet data, holding out webcam sessions."""
import argparse
import logging
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import settings
from src.training.adaptation import adapt_model

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--imported',type=Path,default=settings.ALPHABET_DATASET_PATH)
    p.add_argument('--webcam',type=Path,default=settings.PROJECT_ROOT/'data/processed/alphabet_webcam.csv')
    p.add_argument('--output-dir',type=Path,default=settings.PROJECT_ROOT/'models/alphabet_adapted_v1')
    p.add_argument('--baseline',type=Path,default=settings.PROJECT_ROOT/'models/alphabet_baseline')
    p.add_argument('--canonical-single-hand',action='store_true',help='Experimental right-to-left reflected single-hand features.')
    a=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(levelname)s - %(message)s')
    try:
        adapt_model(a.imported,a.webcam,a.output_dir,a.baseline,canonical=a.canonical_single_hand)
        return 0
    except (OSError,ValueError,RuntimeError) as error:
        logging.error('Adaptation failed: %s',error)
        return 1
if __name__=='__main__': raise SystemExit(main())
