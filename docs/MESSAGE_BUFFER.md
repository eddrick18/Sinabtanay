# Phase 11 — Message construction

Run from the project folder:

```powershell
.\.venv\Scripts\python.exe scripts\run_live.py
```

The default model is now alphabet_single_hand_v3, the latest version checked by the user. --model-dir still selects other compatible models. No models were retrained or overwritten for this phase.

Click the preview to give it keyboard focus. Make a supported letter, wait until Letter shows the correct stable label, then tap Enter to add it. Do not add it if the displayed prediction is wrong: Enter confirms the displayed letter, not your intended sign. Add repeated letters by tapping Enter separately for each one. Holding a key may generate OS repeat events, so tap and release.

Controls:

- Enter: append the current stable letter. Blank, uncertain and no-hand frames cannot add a letter.
- Space: add one word space. Leading and consecutive spaces are skipped.
- Backspace: delete the last character, including a space.
- C: clear the entire message. To add letter C, sign C and press Enter.
- Q: quit.

The footer beneath the camera shows the message, controls, feedback and character count. It avoids hiding any additional hand area. Messages are limited to 200 characters. When a message exceeds the display width, the footer shows its tail with an ellipsis; earlier characters remain in the buffer. Backspace and C still operate on the entire message.

This constructs fingerspelled text with manual confirmation. It does not translate FSL grammar or recognize words automatically. J and Z remain unsupported. Use a test word such as CAT: sign C and Enter, sign A and Enter, sign T and Enter. Check spacing, deletion and clearing. With no hand visible, Enter should leave the message unchanged.

No message, camera image or video is written to disk or sent elsewhere. Closing the program discards the message. Copy/export and speech are not part of this phase.

Implementation: src/recognition/message_buffer.py owns explicit editing and length rules. scripts/run_live.py passes only the current stable label into it after reading the key, then draws the updated message on the next frame. Tests verify explicit confirmation, uncertain label rejection, repeated letters, spacing, deletion, clearing and the length limit; the full suite also simulates camera cleanup. Real keyboard/display behavior remains to be checked in the user's webcam session.
