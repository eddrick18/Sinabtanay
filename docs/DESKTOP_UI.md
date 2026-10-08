# Phase 13 — Desktop interface

Launch the new interface from the project folder:

```powershell
.\.venv\Scripts\python.exe scripts\run_app.py
```

The original run_live.py remains available. Both use the same v3 model, feature processing, confidence filtering, message rules and local speech. No dataset or model was changed for the UI.

## Layout and controls

A dark desktop window shows the live camera on the left and the recognized letter, current model score, message and action buttons on the right. Hand count is displayed under the preview. Add letter is disabled while no stable label is available. Click Add letter only when the displayed letter is correct. Space, Delete and Clear edit the message; Speak reads the current text and Stop speech cancels playback.

Keyboard shortcuts remain Enter, Space, Backspace, C, T, Esc and Q. Focus the application first; tap keys rather than holding them. The message area is read-only and wraps text, with a 200-character limit. Repeated letters require separate confirmations. Recognition and message feedback are displayed separately from speech state. Scores are model scores, not measured accuracy.

The window is resizable with a minimum size of 940x650. The camera preview fits its available space without distorting its aspect ratio. The message area grows with the layout. It does not overlay buttons or text on the camera image.

Options: --camera-index 1 selects another camera; --threshold changes the score filter; --model-dir selects another compatible model; --no-speech disables speech; --debug includes diagnostic tracebacks. Startup failures print an explanation in the terminal. If camera frames stop during use, the displayed letter and camera preview clear, Add letter disables, and message editing remains available. Close and reopen after resolving the camera issue. Q or the window close button releases the camera/detector and stops speech.

## First local test

Close other camera previews, run the app, and check that the camera appears. Construct CAT using the Add letter button, insert a space, delete it, speak the message and stop playback. Try keyboard controls as well. Remove your hand: Add letter should disable. Close the window and check that it releases the webcam. Report any text clipping or layout changes you prefer.

This is still a practice static alphabet prototype, not a full FSL translator. J and Z are unsupported. Images, messages and audio are not saved or sent to a service.

## Implementation and validation

src/ui/app.py uses Python's bundled Tkinter, OpenCV frames and Tk PhotoImage through in-memory PPM encoding. No new package is required. Tk's scheduled callbacks keep the UI updated; speech remains a separate hidden child process. scripts/run_app.py provides the entry point. Existing recognition, message and speech modules remain shared.

All 74 tests pass. UI tests instantiate a withdrawn Tk window, render simulated camera frames, exercise button/message/keyboard actions, and check stale-prediction clearing and resource cleanup on camera failure. Actual visible layout and webcam/audio behavior need the user's local check. The styling is an initial version that can be adjusted based on feedback.

## Sinabtanay redesign
The app is now Sinabtanay. Warm ivory background, forest recognition card, citron primary action, custom s monogram, editorial typography and separate sign/letter/words sections replace the original dark layout. Default window is 1240x820 with minimum 1040x800. Existing model, buttons, shortcuts, camera and speech remain functional. Layout bounds are checked at both sizes; camera/message widget rendering is covered by the regression suite. The repository directory is retained for existing commands.
