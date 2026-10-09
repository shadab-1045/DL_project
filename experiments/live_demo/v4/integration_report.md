# V4 Live Pipeline Integration Report (Phase 7F.17)

## 1. Files Changed
To strictly preserve the isolated V4 environment without destabilizing existing code, the integration was achieved by creating a dedicated V4 module and cleanly swapping it into the API server:
- **Created**: `src/runtime/live_pipeline_v4.py` (V4-specific inference wrapper).
- **Created**: `scratch/test_live_v4.py` (Offline smoke testing suite).
- **Modified**: `src/api/server.py` (Redirected `LiveInferencePipeline` to `LiveInferencePipelineV4`).

## 2. Frozen Checkpoints Used
The system utilizes the immutable V4 checkpoints exactly as evaluated on the held-out test set:
- **Visual Branch & Identity Calibration**: Implicitly loaded via the unified final fusion state dictionary.
- **2D Fusion Layer**: Loaded from `experiments/model_c_v4/best_fusion_model.pt` (Hash: `f2647012bc2863d988e65f980310cac9b60a0beb8c3e068f38b564f4383be2f5`).

## 3. Exact Inference Path
The live pipeline implements the following strictly serialized execution flow for each frame:

1. **Browser Webcam Frame (BGR)**
2. **Native InSwapper Face Swap** (Condition dependent; simulated live attack)
3. **Face Detection & Alignment** (via InsightFace engine)
4. **Model A Feature Extraction** (Frozen ArcFace -> 512d)
5. **Model A Gallery Matching** -> Returns scalar `Cosine Similarity`
6. **V4 Visual Preprocessing** -> 112x112 RGB Crop -> Resize to 224x224 RGB Tensor -> Normalize
7. **Frozen V4 Forward Pass**:
   - `face_tensor` → *Visual Branch* → `P(Synthetic)`
   - `cosine_similarity` → *Identity Branch* → `P(Identity_Match)`
   - `[P(Synthetic), P(Identity_Match)]` → *Fusion Layer* → Raw Logits
8. **Temporal Smoothing (EMA)** -> Smoothed Class Probabilities
9. **Final UI Decision**

*Note: The exact same post-swap frame is utilized for the browser display, Model A identity extraction, and the V4 visual inference.*

## 4. State Mapping
V4's 3-class semantic output is cleanly mapped to the React frontend's established state machine:
- Class 0 (`Genuine`) → `VERIFIED`
- Class 1 (`Different Person`) → `UNKNOWN` (also triggered implicitly if Model A rejects the face)
- Class 2 (`Impersonation`) → `SUSPECTED_IMPERSONATION`

## 5. Latency and Performance (CPU Baseline)
During the offline smoke test (executed on CPU execution providers), the latency profile was measured as follows:
- **Face Detection & Alignment**: ~3.67 seconds
- **Face Swap (Native InSwapper)**: ~6.67 seconds
- **ArcFace Inference**: ~0.08 seconds
- **V4 Visual Branch Inference**: ~0.16 seconds
- **V4 Fusion Inference**: ~0.12 seconds
- **Total Effective Latency (Non-Swapped)**: ~5.34 seconds (0.2 FPS)
- **Total Effective Latency (Swapped)**: ~11.00 seconds

*Note: The V4 overhead (0.28s) is negligible compared to the detection and swap stages. GPU execution is strictly required for real-time live performance.*

## 6. Automated Offline Validation Results
The `scratch/test_live_v4.py` smoke test executed successfully. 
- **Genuine Test**: Successfully passed through the entire V4 tensor stack.
- **Different Person Test**: Correctly triggered the Model A fallback path (mapping to `UNKNOWN` and resetting the EMA state).
- **Impersonation Test**: Correctly applied the Native InSwapper live swap and passed the synthesized frame into the V4 visual branch, successfully extracting a `1.0` identity match and a `0.9998` synthetic probability.

## 7. Deviations from Frozen Pipeline
**None.** The V4 quantitative model architecture, thresholds, parameters, and inputs remained strictly unmodified. Temporal EMA smoothing was implemented strictly at the presentation (UI logic) layer and does not alter the underlying V4 raw feature extraction or predictions.

## 8. Status
**Physical webcam validation is still pending.** The software infrastructure is complete, integrated, and verified to be free of runtime crashes. The V4 model is ready for live physical assessment.
