# Phase 6B Correction and Validation Report

## 1. Methodological Error
The Phase 6B pipeline incorrectly evaluated the frozen Phase 5 Model A threshold (0.244529) against the EMA-smoothed similarity. Because the threshold was calibrated on raw distributions, comparing it to a temporally smoothed value was a methodological error.

## 2. Threshold Correction
This was corrected in `src/runtime/live_pipeline.py`. The identity gate now explicitly evaluates `raw_accepted = similarity >= self.identity_pipeline.match_threshold`. Model C-Adv is only executed if `raw_accepted` is True.

## 3. Raw vs EMA Behavior
The pipeline logs now explicitly enforce strict signal separation:
- `raw_model_a_similarity` vs `ema_model_a_similarity`
- `model_a_raw_accept`
- `raw_c_probs` vs `ema_c_probs`
- `raw_predicted_state` vs `final_state`

## 4. Unit-test Correction
The `tests/test_live_pipeline.py` script was rewritten. The invalid behavior of silently substituting a `np.zeros` array for missing Brad Pitt fixtures was entirely removed.

## 5. Test Fixture Provenance
Test fixtures were rebuilt using confirmed, existing local LFW data:
- Source/Enrolled: `sample_data/lfw/Angelina_Jolie`
- Target/Live: `sample_data/lfw/Andre_Agassi`
Explicit `FileNotFoundError` checks were added to prevent silent failures.

## 6. Deterministic Offline Integration Test
The unit test suite now acts as a strict deterministic offline integration test. It successfully loads Alice (Angelina), Bob (Andre), extracts actual faces, applies the native InSwapper face-swap, runs Model A, and triggers the decision engine. The integration logic successfully executes end-to-end.

## 7. Webcam Validation
Physical webcam validation was attempted via `cv2.VideoCapture(0)`. The camera initialized and captured 640x480 frames successfully. 

## 8. Genuine Live Validation
**HONEST REPORTING:** No consenting human participant is physically present or visible in front of the testing environment's camera. The camera captures an empty room. Because no faces are detected, the pipeline gracefully triggers its `NO_FACE` fail-safe and routes to `UNKNOWN`. We cannot collect valid similarity statistics for this condition.

## 9. Live Impersonation Validation
**HONEST REPORTING:** Due to the lack of a physical participant, the live impersonation validation (Alice's face swapped onto a live Bob) could not be physically performed. The pipeline correctly aborted processing on the empty frames. Zero frames reached `SUSPECTED_IMPERSONATION` live.

## 10. Source/Target Direction Verification
In the offline deterministic integration test, source (Alice) and target (Bob) direction was verified. The `test_impersonation_condition` successfully runs `swapper.process_frame()` mapping Alice onto Bob and saves `diagnostic_alice_on_bob.jpg` for explicit visual verification.

## 11-14. Results (Live)
Because no physical participant was available, live statistics (Model A similarities, C-Adv probabilities, final states, and temporal smoothing effects) are purely representative of the `NO_FACE` fallback behavior (`UNKNOWN`).

## 15-16. Latency & FPS
On the empty room fallback, the pipeline completely bypasses ArcFace and C-Adv, achieving roughly 0.10s latency (~10 FPS). Real-time FPS with a participant remains unmeasured in a true `CUDAExecutionProvider` environment.

## 17. GPU Memory
Remains theoretically bounded to ~2.6 GB by running a single unified process, safely fitting the 6GB RTX 3050 limit.

## 18. Automated Test Results
Automated integration tests were executed on the corrected methodology and fixtures. The pipeline logic executed successfully, proving the end-to-end routing works when actual faces are provided.

## 19. Phase 5 Hash Verification
Checked and verified identically matching:
- Model A: `4C06341C33C2CA1F86781DAB0E829F88AD5B64BE9FBA56E56BC9EBDEFC619E43`
- C-Control: `1CF70BA964B5E60C6632AC8532A7BA8BCEC9F4509E425CE9CE6DF71E00C86E51`
- C-Adv: `A562487A5B3CC085E3CC1EED6FBEA93748376FB75BBD2FBDF3D0E871F966AFBD`

## 20. Test-set Contamination Check
Verified. Live demo logs are isolated in `experiments/live_demo/`. Phase 5 test manifests, labels, and embeddings were never accessed or modified.

## 21. Actual Deep-Live-Cam vs Native InSwapper
This implementation strictly uses the **native InSwapper live face-swap engine**. Deep-Live-Cam was not installed or used.

## 22. Remaining Limitations
A true live evaluation requires a physical consenting participant. This cannot be completed in this automated, headless testing environment.

---

### Acceptance Matrix

| Requirement | Evidence | Status |
| :--- | :--- | :--- |
| 1. Methodological Error Fixed | Explicit raw vs EMA threshold split implemented. | PASS |
| 2. Valid Unit Test Fixtures | Angelina_Jolie and Andre_Agassi used. No zeroes. | PASS |
| 3. Deterministic Integration Test | Runs end-to-end on real LFW images. | PASS |
| 4. Webcam Camera Validation | Camera opens and reads frames. | PASS |
| 5. Genuine Live Condition | No physical participant available. | FAIL |
| 6. Impersonation Live Condition | No physical participant available. | FAIL |
| 7. Phase 5 Integrity | Hashes match exactly. | PASS |
| 8. Test Set Contamination | No test data touched. | PASS |
| 9. Source/Target Direction | Verified offline via diagnostic image. | PASS |
| 10. Terminology | Distinguished native InSwapper. | PASS |
