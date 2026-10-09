# Live Impersonation Result Audit

This document details the audit of the first physical impersonation validation run, identifying exactly why the final state resulted in `UNKNOWN`.

## 1. Session File Inspected
- **Log File**: `experiments/live_demo/manual/manual_impersonation_1791021349.jsonl`
- **Total Frames**: 10
- **Frames with Faces Detected**: 10

## 2. Per-frame Model A Similarities
Model A correctly matched the swapped face to the enrolled Alice/Angelina_Jolie identity with extremely high raw similarities, comfortably clearing the `0.244529` threshold.
- Frame 0: `0.767` (Accepted)
- Frame 1: `0.769` (Accepted)
- Frame 2: `0.778` (Accepted)
- Frame 3: `0.779` (Accepted)
- Frame 4: `0.770` (Accepted)
- Frame 5: `0.779` (Accepted)
- Frame 6: `0.765` (Accepted)
- Frame 7: `0.749` (Accepted)
- Frame 8: `0.775` (Accepted)
- Frame 9: `0.782` (Accepted)

## 3. Per-frame C-Adv Probabilities
The exact C-Adv raw probabilities (Class 0: Genuine, Class 1: Different Person, Class 2: Impersonation) were:
- Frame 0: `[0.456, 0.525, 0.018]`
- Frame 1: `[0.449, 0.534, 0.016]`
- Frame 2: `[0.364, 0.620, 0.017]`
- Frame 3: `[0.373, 0.611, 0.015]`
- Frame 4: `[0.403, 0.579, 0.018]`
- Frame 5: `[0.395, 0.588, 0.017]`
- Frame 6: `[0.414, 0.564, 0.022]`
- Frame 7: `[0.396, 0.587, 0.017]`
- Frame 8: `[0.404, 0.580, 0.016]`
- Frame 9: `[0.376, 0.601, 0.023]`

## 4. Per-frame C-Adv Predicted Class
For all 10 frames, C-Adv predicted **Class 1 (different_person)**, as `0.525 - 0.620` was consistently the maximum probability. It severely underpredicted `impersonation` (~1.7%).

## 5. Per-frame Final State
The final state for all 10 frames was exactly **UNKNOWN**.

## 6. Exact Reason for UNKNOWN
The C-Adv model predicted the `different_person` class. According to the strictly verified decision logic in `live_pipeline.py`:
`if smoothed_class == 0: VERIFIED`
`elif smoothed_class == 2: SUSPECTED_IMPERSONATION`
`else: UNKNOWN`
Because `smoothed_class` was 1, it fell into the `else` branch and correctly mapped to `UNKNOWN`. There is no implementation bug in the state transition logic; the `UNKNOWN` result perfectly reflects the C-Adv model's internal prediction.

## 7. Diagnostic Image Inspection Result
- **Image File**: `experiments/live_demo/manual/diagnostic_impersonation.jpg`
- **Face Visible?** Yes.
- **Swap Visibly Applied?** Yes. The facial features are visibly female (Angelina Jolie) rendered onto the male developer participant. 
- **Processing Failure?** No. The swap successfully blended the source features onto the target geometry.

## 8. Source/Target Direction Verification
- **Source**: Enrolled identity (Alice / Angelina Jolie)
- **Target**: Physical Participant (Bob)
- **Result**: Alice's features were correctly swapped onto Bob's face in the live frame. Direction is verified and correct.

## 9. C-Adv Preprocessing Verification
The preprocessing pipeline is strictly identical between training (`model_c_dataset.py`) and live inference (`live_pipeline.py`):
- **Alignment**: Both use `insightface.utils.face_align.norm_crop(img, kps, image_size=112)`.
- **Color Space**: Both provide RGB inputs to `self.transform`. Training converts `Image.open().convert('RGB')`. Live calls `cv2.cvtColor(BGR2RGB)` and wraps it via `transforms.ToPILImage()`.
- **Tensor Shape/Dtype**: Both use `transforms.Resize((224, 224))`, `transforms.ToTensor()`, and `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`, resulting in identical float32 tensors.

## 10. C-Adv Reference Embedding Verification
The C-Adv live inference correctly receives the `claimed_emb_tensor`.
- `IdentityPipeline.enroll` extracts multiple face embeddings, averages them, and importantly, strictly applies L2 normalization (`emb = emb / np.linalg.norm(emb)`) prior to saving.
- This identically mirrors the training set (which loads pre-normalized `gallery/.../embedding.npy`).
- Live inference pulls exactly this gallery embedding without corrupting or replacing it with the probe embedding.

## 11. C-Adv Input-Frame Verification
The C-Adv model receives the **POST-SWAP PROCESSED FRAME**.
In `live_pipeline.py`, the flow is:
`frame_bgr -> swapper.process_frame() -> processed_frame`
`faces = engine.extract_faces(processed_frame)`
`aligned_rgb = align_face(processed_frame, best_face.kps)`
The C-Adv tensor is explicitly derived from `processed_frame`, verifying the synthetic face reaches the classification branch.

## 12. Comparison With Intended Decision Logic
The implementation strictly follows the intended behavior:
- `Model A accepts + C-Adv different_person = UNKNOWN`
The fact that C-Adv confidently predicted `different_person` instead of `impersonation` reveals a model generalization failure, not a code flaw. C-Adv is mistaking the specific synthetic artifacts of `inswapper_128.onnx` applied over a live webcam feed as genuine biometric dissimilarities, rather than synthetic face-swapping.

## 13. Performance Measurements
- **Average Latency**: 5.707 s/frame (across the 10-frame sample).
- **Approximate FPS**: 0.2 FPS.
- **Root Cause**: The execution environment relies on `CPUExecutionProvider` fallback due to missing/unbound CUDA bindings in the Python subprocess. Both `InSwapper` and `ArcFace` sequentially choke the CPU.

## 14. Implementation Discrepancy
There are **zero** implementation discrepancies found. Preprocessing, logic gates, tensors, embeddings, and image states perfectly mirror Phase 5 testing parameters.

## 15. Recommended Next Correction
No code correction is required for the pipeline infrastructure. The UNKNOWN output is a true operational finding demonstrating that the adversarially trained C-Adv model (Phase 5) does not cleanly generalize to the `inswapper_128.onnx` generator in a live webcam domain. It confuses the synthetic presentation as a different real person (preventing false VERIFIED, but failing to trigger SUSPECTED_IMPERSONATION alerts).
The proper scientific response is Phase 6C: potentially retraining or augmenting C-Adv data with native `inswapper` distributions, or recalibrating decision logic thresholds based on domain adaptation techniques. Do not manually hack the pipeline code to force a result.
