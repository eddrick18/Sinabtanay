# Phase 12 — Optional local text-to-speech

Run the same live command:

```powershell
.\.venv\Scripts\python.exe scripts\run_live.py
```

Build a message with Enter and Space. Tap T to speak the current message through the default Windows audio device. Esc stops speech; Q closes the application and stops an active speech process. Enter still adds the recognized letter T: pressing the keyboard T is the speech action. The message remains editable while speaking; edits affect the next request, not the message already sent to the voice. Clearing text does not stop current speech; use Esc.

Speech is optional and only starts on T. Empty messages are rejected. While a message is being spoken, further T requests are ignored rather than queued. Tap and release keys: OS key repeat can retrigger after speech finishes. The footer displays speech status. The webcam stays active during voice startup and playback.

To disable speech:

```powershell
.\.venv\Scripts\python.exe scripts\run_live.py --no-speech
```

The implementation uses Windows System.Speech through a hidden Windows PowerShell child process, with no added Python dependency. Messages are UTF-8/base64 data on stdin, never interpolated into shell commands. No text or audio file is saved and no external speech service is contacted. Installed local voice and audio device determine pronunciation; this is speech of constructed text, not FSL translation. Voice/language selection remains a later interface option.

On this machine Microsoft Zira Desktop is installed and Windows PowerShell execution policy is RemoteSigned. If speech status reports unavailable, check volume/default output device and installed Windows voices. Existing system execution policy is respected; this implementation does not change it. Failure leaves message editing and recognition running.

Files: src/speech/local_speech.py owns asynchronous process lifecycle and controls; src/speech/speak.ps1 loads the local voice and reads message data; run_live.py binds T/Esc and displays state. Automated tests simulate process startup, safe text transport, empty/disabled requests, duplicate requests, completion/failure and cleanup. Installed voice availability was checked without playing audio. Actual playback is for the user's local test.

Test: construct CAT, tap T, confirm the voice says the text while camera recognition continues. Try Esc during a longer message, T with an empty message, and Q during speech. All 71 tests pass.
