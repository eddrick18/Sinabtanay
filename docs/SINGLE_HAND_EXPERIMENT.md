# Single-hand normalization experiment — 2026-10-08

Inventory: imported examples contain 6,287 left-only, 2,926 right-only, and 8 two-hand feature vectors. Imported O has 173 left-only and 113 right-only. Webcam O has 58 two-hand and 2 right-only vectors across 60 samples; there are no left-only O vectors. These are detected handedness slots, not independently verified physical-hand labels.

The experimental transform single-hand-reflect-x-left-v1 leaves left-only features unchanged and reflects normalized right-only x coordinates into the left slot, clearing the right slot. y/z and wrist/scale normalization stay unchanged. Two-hand features remain unchanged. This assumes a mirrored single-hand handshape retains the dataset letter label; it is a modeling hypothesis, not an FSL equivalence claim. Orientation and linguistic correctness still need validation. Existing datasets and models are unchanged. Metadata declares the runtime transform so the same transformation is applied at prediction time.

Run:

```powershell
.\.venv\Scripts\python.exe scripts\run_live.py --model-dir models\alphabet_single_hand_experiment
```

Keep only the signing hand visible. Check Hands detected: 1. Try natural O with left and right hands separately, then other letters. Compare with alphabet_adapted_v2 using the same lighting and poses. The threshold remains 0.70. Do not contort the sign to increase its score.

On the existing 80 held-out A/O samples, v2 scored 98.75% and this experiment scored 100%. The O recordings are dominated by two-hand detections, so this does not prove a right-hand O improvement. These recordings were already examined in earlier model comparisons; they now serve development evaluation, not a fresh final accuracy estimate. A new one-hand-per-attempt webcam check is necessary before recommending replacement. The baseline model remains available.

The live preview now displays hand count. Tests cover mirrored pair equivalence, idempotence, unchanged two-hand inputs, invalid feature rejection and real experimental model loading. The full regression suite uses a writable workspace temporary directory because the Windows sandbox's default temporary directory rejected writes.
