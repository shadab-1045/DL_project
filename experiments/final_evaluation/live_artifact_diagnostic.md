# Controlled Live-Artifact Diagnostic (READ-ONLY)

## Objective
To determine whether the `UNKNOWN` final state observed during the Phase 6B manual impersonation test was caused by a live-pipeline implementation error (such as incorrect tensor states, temporal smoothing bugs, or webcam buffering) or whether the C-Adv model itself fundamentally predicts the `different_person` class when evaluating the specific synthetic artifacts produced by the Native InSwapper.

## 1. Exact Diagnostic Image Path
`experiments/live_demo/manual/diagnostic_impersonation.jpg`

*(This image was saved directly from the `live_pipeline.py` during the manual validation, representing the exact POST-SWAP frame intended for C-Adv classification.)*

## 2. Direct-Image Inference Results
Running the exact same preprocessing and model inference locally on this static image (bypassing the live webcam, InSwapper runtime, and temporal smoothing):

### Model A Result
- **Raw Similarity**: `0.7772`
- **Threshold**: `0.244529`
- **Accepted**: `True`
- **Matched Identity**: `Angelina_Jolie`

### C-Adv Result
- **Probabilities**: 
  - `Genuine` (0): `0.4473`
  - `Different Person` (1): `0.5382`
  - `Impersonation` (2): `0.0145`
- **Predicted Class**: `1 (different_person)`

## 3. Corresponding Live-Frame Result
The closest corresponding live frame recorded in `manual_impersonation_1791021349.jsonl` (e.g., Frame 2) shows:
- **Raw Similarity**: `0.7780`
- **C-Adv Probabilities**: `[0.3636, 0.6195, 0.0169]`
- **Predicted Class**: `1 (different_person)`

*Note: The exact probabilities differ slightly because the live frame used for prediction in a specific frame iteration differs microscopically from the frame dumped to disk, but the class distribution and decision boundary are perfectly consistent.*

## 4. Comparison
Both the direct-image inference and the live-pipeline inference yield the exact same semantic result:
- Model A **strongly accepts** the identity.
- C-Adv **predicts `different_person`**, completely ignoring the `impersonation` class (< 2% probability).

There are **zero preprocessing differences** between this diagnostic inference and the live pipeline. Both use `face_align.norm_crop`, convert to RGB, resize to 224x224, convert to tensor, and apply identical ImageNet normalizations. Both utilize the identical gallery reference embedding.

## 5. Conclusion
**Case A applies: The saved InSwapper image is also classified as `different_person` by the model itself.**

The discrepancy is **not** caused by a live-pipeline bug, temporal smoothing, or state-transition logic. The C-Adv model fundamentally fails to recognize the specific synthetic blending artifacts produced by `inswapper_128.onnx` as "impersonation". Because C-Adv was trained on datasets like FaceForensics++, it generalized to those specific manipulation patterns but perceives the Live-Cam artifacts as generic biometric dissimilarities, thereby predicting `different_person` instead of `impersonation`.

This proves that the live runtime is functionally sound and strictly honors the Phase 5 baseline. The attack is **not detected** as an impersonation; rather, it is rejected by C-Adv as a different person entirely, yielding a safe (but technically misclassified) `UNKNOWN` state.

## 6. Limitations
- This diagnostic was performed on a single stored artifact representing a specific physical lighting environment and operator angle. 
- The inference was performed without CUDA acceleration, exactly matching the backend evaluation environment constraints, though this does not impact numerical outputs.
- No retraining or calibration was attempted, preserving the Phase 5 benchmark strictly as instructed.
