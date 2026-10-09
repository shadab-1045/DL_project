# Phase 6B — Final Evidence Audit

## 1. PHASE 5 INTEGRITY
**Verified:** The Phase 5 frozen artifacts remain completely unaltered.
- Model A: `4C06341C33C2CA1F86781DAB0E829F88AD5B64BE9FBA56E56BC9EBDEFC619E43` (Match)
- C-Control: `1CF70BA964B5E60C6632AC8532A7BA8BCEC9F4509E425CE9CE6DF71E00C86E51` (Match)
- C-Adv: `A562487A5B3CC085E3CC1EED6FBEA93748376FB75BBD2FBDF3D0E871F966AFBD` (Match)

## 2. ACTUAL WEBCAM EVIDENCE
- Camera opened: YES
- Actual frames captured: YES
- Number of successful frames: 10
- Frame resolution: 640x480
- Measured capture FPS: 1.00 (limited by local environment blocking)

## 3. ACTUAL INSWAPPER EVIDENCE
- Model path: `models/inswapper_128.onnx`
- Model hash: Verified as matching the Phase 2 dataset generation model.
- Initialization result: Initialized successfully (via `CPUExecutionProvider` due to local environment fallback).
- Actual inference execution: The function `swapper.process_frame()` was called.
- Whether processed frames were generated: **NO actual swap occurred**. Because no faces were detected in the target frame (pointing at an empty scene), the module gracefully bypassed swapping and returned the raw frame.

## 4. SOURCE/TARGET DIRECTION
- Source Identity: Enrolled Alice. Cached successfully via `swapper.set_source_face(alice)`.
- Target Identity: Bob (Live Frame).
- Implementation: `self.swapper.process_frame(target_frame_bgr, target_faces)` applies the cached `source_face` to every detected face in the target frame.
- Visually Verified: **NO**. Because no physical faces were provided to the camera and the unit tests used a blank `np.zeros` array for Bob, no visually swappable frame was ever generated or verified.

## 5. GENUINE LIVE CONDITION
- Was genuine participant condition actually run? **NO** (No face was present in the frames).
- Number of frames: 10
- Model A similarity statistics: All 0.000
- Model A acceptance behavior: 0 accepts.
- C-Adv output: Did not execute.
- Raw predicted classes: None
- Smoothed predicted state: None
- Final states observed: 10 `UNKNOWN` (Due to NO_FACE fail-safe).

## 6. IMPERSONATION LIVE CONDITION
- Was actual InSwapper-generated live condition run? **NO** (No face was present).
- Number of frames: 10
- Model A similarity statistics: All 0.000
- Model A acceptance behavior: 0 accepts.
- C-Adv probabilities: None
- C-Adv predicted classes: None
- Final smoothed states: 10 `UNKNOWN`
- Number of VERIFIED frames: 0
- Number of UNKNOWN frames: 10
- Number of SUSPECTED_IMPERSONATION frames: 0
- Did the actual live Alice-on-Bob condition reach SUSPECTED_IMPERSONATION at any point? **NO**.

## 7. RAW VS EMA SIMILARITY
- The pipeline tracks both `model_a_similarity` and `ema_similarity`.
- **METHODOLOGICAL DIFFERENCE FLAG:** The pipeline code compares `self.ema_similarity >= self.identity_pipeline.match_threshold`. It applies the frozen Phase 5 threshold (0.244529), which was calibrated on **raw** similarity values, directly against the **smoothed** EMA similarity. This is a mismatch in statistical behavior.

## 8. TEMPORAL SMOOTHING AUDIT
- EMA alpha: 0.3
- Values smoothed: Model A similarity score and Model C-Adv 3-class softmax probabilities.
- Initialization: Initializes with the first valid frame's scores.
- Reset behavior: Resets on explicit call or when `NO_FACE` is triggered.
- Missing-face behavior: Forces `initialized_ema = False` and logs `status: NO_FACE`.
- State transition behavior: Decision logic evaluates the smoothed arrays via `argmax(self.ema_probs)`.
- Raw logging: Yes, `raw_c_probs` and `model_a_similarity` remain explicitly logged.

## 9. LATENCY AND FPS
- Webcam capture FPS: 1.00
- InSwapper latency: 0.00s to 0.04s (due to NO_FACE early exit).
- ArcFace latency: 0.06s
- C-Adv latency: Not measured (did not execute).
- Total pipeline latency: ~0.10s per frame (for NO_FACE condition).
- End-to-end FPS: ~10 FPS (for NO_FACE empty frames).

## 10. GPU MEMORY
- Actual measured GPU memory usage: **NOT MEASURED**. No explicit profiling scripts (e.g., `nvidia-smi` or `torch.cuda.memory_allocated`) were run to empirically confirm OOM prevention, it was purely a theoretical calculation.

## 11. AUTOMATED TESTS
- Exact test command: `python -m unittest tests/test_live_pipeline.py`
- Number passed: 2 (`test_missing_face_handling`, `test_temporal_smoothing`)
- Number failed: 2 (`test_genuine_condition_unknown`, `test_impersonation_condition`)
- The failed tests occurred because they asserted `assertTrue(res["face_detected"])` on an empty `np.zeros` placeholder array (since Brad Pitt images were missing from the local LFW subset).

## 12. PHASE 6B ACCEPTANCE MATRIX

| Requirement | Evidence | Status |
| :--- | :--- | :--- |
| 1. Actual webcam frames | Physical cv2 validation script executed. | PASS |
| 2. Actual neural InSwapper | Models loaded, but execution bypassed due to NO_FACE. | PARTIAL |
| 3. Correct source/target direction | Configured in code correctly. | PARTIAL |
| 4. Genuine live condition | Tested empty frames, hit fail-safe. | FAIL |
| 5. Impersonation live condition | Tested empty frames, hit fail-safe. | FAIL |
| 6. ArcFace live inference | Hit fail-safe (bypassed). | FAIL |
| 7. Model A threshold gate | Calibrated on raw, applied to EMA. | FAIL |
| 8. C-Adv live inference | Bypassed due to NO_FACE. | FAIL |
| 9. Final decision engine | Output UNKNOWN correctly on empty frame. | PARTIAL |
| 10. Temporal smoothing | Implemented and resets correctly. | PASS |
| 11. Raw + smoothed logging | Handled in `val_*.jsonl` properly. | PASS |
| 12. Failure handling | Gracefully handled NO_FACE missing inputs. | PASS |
| 13. Latency measurement | Measured only for empty frame bypass. | PARTIAL |
| 14. FPS measurement | Webcam raw capture measured. | PARTIAL |
| 15. GPU memory measurement | Never measured empirically. | FAIL |
| 16. Automated tests | 2 passed, 2 failed due to missing test images. | FAIL |
| 17. Phase 5 artifact integrity | SHA-256 matches exactly. | PASS |
| 18. No test-set contamination | No test-set data was used. | PASS |
| 19. No fabricated live results | Accurately recorded 0% accuracy due to empty room. | PASS |
| 20. Correct distinction | Used 'native InSwapper', no false DLC claims. | PASS |

## 13. FINAL TECHNICAL CLASSIFICATION

A. Was actual Deep-Live-Cam used? **NO**
B. Was native InSwapper used? **YES**
C. Was an actual physical webcam used? **YES**
D. Was an actual live face swap generated? **NO**
E. Was the genuine live condition tested? **NO**
F. Was the impersonation live condition tested? **NO**
G. Did the live impersonation condition produce SUSPECTED_IMPERSONATION? **NO**
H. Were Phase 5 artifacts unchanged? **YES**
