# Manual Live Validation Protocol

## 1. Equipment Requirements
- A developer workstation with the Python environment successfully configured (e.g., `venv` active).
- A physical webcam accessible via `cv2.VideoCapture(0)`.
- GPU (optional, but highly recommended, specifically requiring `CUDAExecutionProvider` to achieve real-time >5 FPS).

## 2. Camera Setup
- Ensure the webcam is unobstructed, sufficiently lit, and pointing at the area where the participant will sit.
- The pipeline inherently runs single-face matching; ensure only one face is predominantly visible.

## 3. Enrolled Identity Setup
By default, the script enrolls **Alice** using images from `sample_data/lfw/Angelina_Jolie`.
- This sets the InSwapper **Source Face** to Alice.
- The ArcFace **Gallery** will hold Alice's identity embeddings.

## 4. Genuine Procedure (G)
1. The physical participant must be the enrolled person (or represent the enrolled person for the sake of the test).
2. Press `G` to enter **Genuine** mode.
3. No face swap is applied. The pipeline processes raw camera frames.
4. **Expected:** `Model A raw similarity >= 0.244529`, C-Adv predicts `Genuine (0)`, final state returns `VERIFIED`.

## 5. Impersonation Procedure (I)
1. A physical participant who is **NOT** the enrolled identity (e.g., Bob) sits in front of the camera.
2. Press `I` to enter **Impersonation** mode.
3. The script utilizes the **native InSwapper** to apply the source identity (Alice) onto the live physical participant (Bob).
4. **Expected:** Model A accepts the identity geometry, but C-Adv detects the synthetic blend. Final state returns `SUSPECTED_IMPERSONATION`.

## 6. Expected Pipeline
`Raw Webcam Frame -> Native InSwapper (if 'I' pressed) -> Face Alignment -> ArcFace Embedding -> Raw Model A Gate (Threshold: 0.244529) -> C-Adv Inference -> Temporal Smoothing (EMA) -> Final Output`

## 7. What to Observe
The `cv2.imshow` window provides an explicit visual HUD displaying:
- **Visuals:** The actual swapped/unswapped frame confirming source/target direction.
- **FPS:** The current frame rate.
- **Raw Sim / Threshold:** Strict identity thresholding behavior.
- **C-Adv Probabilities:** 3-class distribution outputs in real-time.
- **Final State:** The temporally smoothed user-facing decision.

## 8. What Gets Logged
All sessions generate `.jsonl` files stored distinctly outside the Phase 5 benchmark:
`experiments/live_demo/manual/manual_<condition>_<timestamp>.jsonl`
Logs record: `timestamp`, `frame_id`, `raw_model_a_similarity`, `model_a_threshold`, `model_a_raw_accept`, `ema_model_a_similarity`, `raw_c_probs`, `final_state`, `latency`, `fps`.

## 9. How to Terminate the Session
Press the **[Q]** key with the `cv2.imshow` window actively focused.
A terminal summary will automatically print metrics for the session.

## 10. Privacy Considerations
- No live webcam frames are saved to disk during this diagnostic session. 
- The operator must explicitly consent to being the physical target.
- Logs capture biometric probability values but do not store physical imagery.

## 11. Distinction between Live Validation and Phase 5 Benchmark
- **Phase 5 Benchmark** is frozen, immutable, and purely evaluates offline datasets.
- **Manual Live Validation** dynamically verifies real-time hardware ingestion and temporal smoothing behavior. These results are inherently subjective to the physical lighting and operator and are completely walled-off from Phase 5 scientific metrics.

## Pre-flight Checks
The `manual_live_validation.py` script automatically verifies the `SHA-256` hashes of Model A, C-Control, and C-Adv against the Phase 5 baseline before starting. It will explicitly block execution if corruption is detected.
