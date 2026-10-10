# Limitations

A short, plain list of what this project does not do. Each item links to the evidence.

## Model

1. **Not robust to a white-box attacker.** A perturbation of 1/255 per pixel is enough to reach 0% detection. Adversarial training was tried and made the detector unusable (62–67% of real faces blocked). See [Evaluation](evaluation.md#3-a-white-box-attacker-defeats-the-detector).
2. **One swapper.** Training and test data come from InSwapper only. Behaviour on other face-swap methods was not measured.
3. **Small, narrow data.** 1,886 + 1,886 training crops, 263 + 263 test crops, from LFW and FaceForensics++. Differences of a few percentage points are within noise. No confidence intervals are reported for the deployed model.
4. **Digital test data.** Test crops are image files processed by the same pipeline that made the training data. Real cameras add compression, exposure and motion blur the detector may not have seen. Training used degradations (JPEG, blur, resize, brightness, noise) to reduce this gap, but the gap was not measured.
5. **Threshold tuned on one user.** The deployed threshold of 0.3 was chosen with six webcam frames of one person in view. It is calibrated for that setup, not proven in general.

## System

6. **Per-frame rates are not session rates.** 6.1% of real frames flagged does not tell you how often a real user is blocked over a session of 8-frame windows. That was not measured.
7. **No liveness.** A replayed video or a photo of the enrolled user is not a swap and is not addressed.
8. **ArcFace is trusted for identity.** It was not attacked in the white-box tests.
9. **FaceGuard is not part of the verdict.** The bundled Keras classifier accepts only about 31% of real faces as real on FaceForensics++ and is trivially evaded; it exists for the legacy tabs.
10. **No authentication.** See [Security](security.md).

## Repository

11. **Stale comments.** `swapdet.py` and `unified_api.py` describe the detector as "adversarially trained". The deployed weights are the standard-trained checkpoint.
12. **Historical research folder.** `DL_project/experiments/` and `DL_project/docs/` hold results from earlier phases, including a "Model C" that is not in the live verdict path. Their summaries were not re-verified here and should not be quoted without checking them against the code.
13. **Not reproducible from a bare clone.** Datasets, FaceForensics++ frames and most model weights are git-ignored and must be obtained or regenerated.
14. **No container setup in this folder.** The standalone repository has Docker files; they are not included here because they were not re-verified.

## What would come next

- Evaluate against other swap methods, and train on a mix.
- Evaluate on real webcam captures from several people, with confidence intervals.
- Measure session-level false-block rate for the 8-frame window.
- Investigate robustness approaches that do not collapse clean accuracy (the failures here suggest the training recipe, not just the idea, needs work).
- Add authentication and an OTLP exporter if the API is ever deployed beyond a laptop.
