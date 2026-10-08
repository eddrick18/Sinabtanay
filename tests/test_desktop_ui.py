import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock,patch
import numpy as np
from config import settings
from src.ui.app import SignBridgeApp

@unittest.skipUnless((settings.PROJECT_ROOT / "models/alphabet_single_hand_v3/trained/sign_classifier.joblib").is_file(), "Local v3 model required for desktop integration checks.")
class DesktopTests(unittest.TestCase):
    def setUp(self):
        self.root=tk.Tk(); self.root.withdraw()
        self.camera=MagicMock(); self.camera.read.return_value=(True,np.zeros((480,640,3),dtype=np.uint8))
        self.detector=MagicMock(); self.detector.detect.return_value=SimpleNamespace(hand_landmarks=[])
        self.camera_patch=patch('src.ui.app.cv2.VideoCapture',return_value=self.camera)
        self.detector_patch=patch('src.ui.app.HandDetector',return_value=self.detector)
        self.draw_patch=patch('src.ui.app.draw_hand_results')
        for p in [self.camera_patch,self.detector_patch,self.draw_patch]: p.start(); self.addCleanup(p.stop)
        self.app=SignBridgeApp(self.root,settings.PROJECT_ROOT/'models/alphabet_single_hand_v3',speech_enabled=False)
        self.addCleanup(self.app.close)

    def test_render_no_hand_and_button_edits(self):
        self.root.after_cancel(self.app.after_id); self.app.after_id=None
        self.app.tick()
        self.assertEqual(self.app.letter.cget('text'),'--')
        self.assertEqual(self.app.buttons['Add letter'].cget('state'),'disabled')
        self.assertIsNotNone(self.app.photo)
        self.app.action(13); self.assertEqual(self.app.message.text,'')
        self.app.current_label='C'; self.app.action(13)
        self.app.current_label='A'; self.app.action(13)
        self.app.current_label='T'; self.app.action(13)
        self.assertEqual(self.app.message.text,'CAT')
        self.assertEqual(self.app.message_view.get('1.0','end-1c'),'CAT')
        self.app.action(32); self.app.action(8); self.assertEqual(self.app.message.text,'CAT')
        self.app.action(ord('c')); self.assertEqual(self.app.message.text,'')

    def test_failure_clears_prediction_and_cleanup(self):
        self.root.after_cancel(self.app.after_id); self.app.after_id=None
        self.app.current_label='A'; self.camera.read.return_value=(False,None)
        self.app.tick(); self.assertIsNone(self.app.current_label)
        self.app.action(13); self.assertEqual(self.app.message.text,'')
        self.camera.release.assert_called_once(); self.detector.close.assert_called_once()
        self.app.close(); self.app.close()
        self.camera.release.assert_called_once()

    def test_keyboard_speech_actions(self):
        self.app.current_label='A'
        self.app.keypress(SimpleNamespace(keysym='Return',char='\r'))
        self.assertEqual(self.app.message.text,'A')
        self.app.speech=MagicMock(); self.app.speech.speak.return_value='Speaking'
        self.app.buttons['Speak'].invoke() # disabled speech button does nothing
        self.app.keypress(SimpleNamespace(keysym='t',char='t'))
        self.app.speech.speak.assert_called_once_with('A')
        self.app.keypress(SimpleNamespace(keysym='Escape',char=''))
        self.app.speech.stop.assert_called_once()

if __name__=='__main__': unittest.main()
