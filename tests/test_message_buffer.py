import unittest
from src.recognition.message_buffer import MessageBuffer

class MessageTests(unittest.TestCase):
    def test_explicit_add_repeat_and_uncertain(self):
        b=MessageBuffer(['A','B'])
        b.handle_key(255,'A'); self.assertEqual(b.text,'')
        b.handle_key(13,None); self.assertEqual(b.text,'')
        b.handle_key(13,'A'); b.handle_key(10,'A')
        self.assertEqual(b.text,'AA')
        b.handle_key(13,'Z'); self.assertEqual(b.text,'AA')

    def test_edit_spacing_and_limit(self):
        b=MessageBuffer(['A'],limit=3)
        b.handle_key(32,None); self.assertEqual(b.text,'')
        b.handle_key(13,'A'); b.handle_key(32,None); b.handle_key(32,None)
        self.assertEqual(b.text,'A ')
        b.handle_key(13,'A'); b.handle_key(13,'A')
        self.assertEqual(b.text,'A A')
        b.handle_key(8,None); self.assertEqual(b.text,'A ')
        b.handle_key(127,None); self.assertEqual(b.text,'A')
        b.handle_key(ord('c'),None); self.assertEqual(b.text,'')
        b.handle_key(8,None); self.assertEqual(b.text,'')

if __name__=='__main__': unittest.main()
