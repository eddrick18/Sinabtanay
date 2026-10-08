"""Sinabtanay desktop interface: camera, message editing and optional local speech."""
import argparse
import logging
from pathlib import Path
import sys
import tkinter as tk
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import settings
from src.ui.app import SinabtanayApp

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,default=settings.PROJECT_ROOT/'models/alphabet_single_hand_v3')
    p.add_argument('--camera-index',type=int,default=settings.CAMERA_INDEX)
    p.add_argument('--threshold',type=float,default=.70)
    p.add_argument('--no-speech',action='store_true')
    p.add_argument('--debug',action='store_true')
    a=p.parse_args(); root=None
    try:
        root=tk.Tk()
        app=SinabtanayApp(root,a.model_dir,a.camera_index,a.threshold,not a.no_speech)
        root.mainloop()
        return 0
    except (OSError,ValueError,RuntimeError,tk.TclError,cv2.error) as error:
        logging.error('Application failed: %s',error,exc_info=a.debug)
        if root is not None:
            try: root.destroy()
            except tk.TclError: pass
        return 1
if __name__=='__main__': raise SystemExit(main())
