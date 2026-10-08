# Live alphabet prediction

From the signbridge-fsl folder:

```powershell
.\.venv\Scripts\python.exe scripts\run_live.py
```

The default model is models/alphabet_baseline. Close other camera applications first. Focus the preview and press Q to quit. If camera 0 cannot open, try --camera-index 1. Add --debug to show FPS and diagnostic errors.

Use a well-lit background and keep your hand fully visible. Hold a known dataset alphabet handshape steady. The preview displays Letter, status, and the current model score. A score is a classifier output, not a calibrated probability of linguistic correctness. No images or video are saved.

The default threshold is 0.70. Five agreeing predictions within seven recent qualifying frames are required. Low confidence, no hand, or ambiguous landmarks clear the history immediately. A changed current letter suppresses the old label while it settles. These are initial settings to evaluate on fresh webcam examples, not validated accuracy thresholds.

The pipeline loads metadata and verifies its preprocessing contract, installed scikit-learn version and model checksum before loading the local joblib artifact. It mirrors frames consistently with training, detects landmarks in VIDEO mode, normalizes the same 126 features, predicts class scores, filters predictions and draws the display. Use --model-dir to select another compatible locally trained run. Only load model artifacts you trust.

This default is a practice model for 24 static letters. J and Z are excluded. It does not assemble words or translate sentences. Test several letters and both hands; the imported data may not generalize to your lighting, orientation or camera. Wrong high-score predictions need fresh representative data and evaluation, not just a lower threshold.

Implementation: scripts/run_live.py owns the webcam and cleanup; src/recognition/predictor.py owns model checks and temporal filtering. Automated checks load the actual saved model, verify filtering and contract rejection, and simulate camera/no-hand handling and cleanup. A real webcam session must be checked locally by the user.
