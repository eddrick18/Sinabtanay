import base64
import unittest
from unittest.mock import MagicMock,patch
from src.speech.local_speech import LocalSpeech

class SpeechTests(unittest.TestCase):
    def test_empty_disabled_and_limit_do_not_launch(self):
        with patch('src.speech.local_speech.subprocess.Popen') as spawn:
            self.assertIn('empty',LocalSpeech().speak(' '))
            self.assertIn('disabled',LocalSpeech(False).speak('CAT'))
            self.assertIn('long',LocalSpeech().speak('A'*201))
            spawn.assert_not_called()

    def test_nonblocking_safe_message_and_busy(self):
        child=MagicMock(); child.poll.return_value=None
        with patch('src.speech.local_speech.subprocess.Popen',return_value=child) as spawn, patch('src.speech.local_speech.Path.is_file',return_value=True):
            s=LocalSpeech(); text='CAT; $notCode'
            self.assertIn('Speaking',s.speak(text))
            child.stdin.write.assert_called_once_with(base64.b64encode(text.encode()))
            child.stdin.close.assert_called_once()
            self.assertNotIn(text,spawn.call_args.args[0])
            child.wait.assert_not_called()
            self.assertIn('Already',s.speak('DOG'))
            self.assertEqual(spawn.call_count,1)
            s.stop(); child.terminate.assert_called_once(); child.wait.assert_called_once()

    def test_completion_and_failure(self):
        s=LocalSpeech(); child=MagicMock(); s.process=child
        child.poll.return_value=0
        self.assertIn('finished',s.poll()); self.assertIsNone(s.process)
        s.process=child; child.poll.return_value=1
        self.assertIn('unavailable',s.poll()); self.assertIsNone(s.process)
        with patch('src.speech.local_speech.subprocess.Popen',side_effect=OSError),patch('src.speech.local_speech.Path.is_file',return_value=True):
            self.assertIn('Could not',s.speak('CAT'))

    def test_close_stops_active_child(self):
        s=LocalSpeech(); child=MagicMock(); child.poll.return_value=None; s.process=child
        s.close(); child.terminate.assert_called_once(); self.assertIsNone(s.process)

if __name__=='__main__': unittest.main()
