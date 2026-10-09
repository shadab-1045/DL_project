# Interactive Enrollment and Live Face-Swap Demo

This document outlines the architecture and workflows for the Phase 7E interactive live demonstration of the identity-aware anti-impersonation system.

## 1. Demo/Research Separation
To ensure the integrity of the frozen research evaluation artifacts (Phase 5 & 6), the interactive demonstration runs in an isolated **Demo Mode**:
- Uses a separate gallery directory: `experiments/live_demo/demo_gallery`.
- Does not modify any frozen test manifests, thresholds (`0.244529`), or evaluation metrics.
- Enrolling a new identity creates temporary data distinct from the rigorous `LFW` or `VGGFace2` research galleries.
- Note: This system is primarily single-session due to the global `LiveInferencePipeline` lock, meaning concurrent users will overwrite each other's active `Native InSwapper` source face.

## 2. Interactive Enrollment Workflow
Users can create a new identity representation (the **SOURCE**) on the fly through the browser:
1. User enters an Identity Name (e.g., "Rahul").
2. User selects 3–5 reference images and uploads them.
3. The FastAPI backend extracts facial embeddings using the frozen `Model A` ArcFace weights.
4. The embeddings are mean-aggregated and L2-normalized.
5. The identity is saved to `demo_gallery` and the best detected face is loaded into memory as the `Native InSwapper` source.

## 3. Terminology: Source and Target
- **SOURCE**: The enrolled identity (the person who is supposedly presenting to the system). This is the face that Native InSwapper will paste.
- **TARGET**: The physical person standing in front of the webcam. This is the body/background the face will be pasted onto.

## 4. Live Face-Swap Verification Path (Face Swap ON)
When the user toggles "FACE SWAP ON", the system simulates a synthetic deepfake presentation attack:
1. The browser sends the live webcam frame (Target) to the backend over WebSocket.
2. The backend extracts the target faces and runs `Native InsightFace InSwapper` to apply the Source face. (Note: InSwapper uses the first uploaded reference image as its source geometry, whereas Model A evaluates the aggregated mean of all 3-5 images).
3. The resulting **POST-SWAP** frame is stored in memory.
4. The exact **POST-SWAP** frame is then fed into the verification pipeline (Model A + C-Adv).
5. The backend encodes the **POST-SWAP** frame to a base64 JPEG string and sends it back to the browser alongside the ML inference metrics.
6. The React frontend overlays the swapped frame on top of the webcam feed, proving that the verification pipeline analyzed the identical image the user is seeing.

## 5. Interpreting the Final State
The UI adheres to strict scientific integrity by presenting exactly what the backend produces:
- **VERIFIED**: Model A accepted the identity, and C-Adv classified it as `genuine`.
- **SUSPECTED_IMPERSONATION**: Model A accepted the identity, but C-Adv successfully flagged it as `impersonation`. This proves the anti-impersonation defense worked against the synthetic attack.
- **UNKNOWN**: The identity was rejected by Model A, OR it was flagged as `different_person` by C-Adv.

## 6. Performance Expectations
Because the inference backend (`onnxruntime` and `torch`) may fall back to CPUExecutionProvider on certain systems, processing a full Face-Swap + ArcFace + C-Adv pipeline can take 1–3 seconds per frame.
- The UI handles this via a conservative capture interval (1500ms) and an `isProcessing` lock to prevent frame backlogs.
- The browser remains responsive and drops intermediate webcam frames until the backend is ready, pacing the experience to the hardware's capabilities.

## 7. Manual Demo Validation Protocol
To validate the system, perform the following tests using the browser UI at `http://localhost:5173`:

### TEST A — GENUINE
1. Enroll "Person A" (yourself) using 3–5 local photos.
2. Stand in front of the webcam.
3. Keep Face Swap **OFF**.
4. Observe the result: The UI should show `VERIFIED`.

### TEST B — DIFFERENT PERSON
1. Keep "Person A" enrolled.
2. Have "Person B" (a friend) stand in front of the webcam.
3. Keep Face Swap **OFF**.
4. Observe the result: Model A should reject Person B, yielding `UNKNOWN`.

### TEST C — SYNTHETIC IMPERSONATION
1. Keep "Person A" enrolled (this is the SOURCE).
2. Have "Person B" stand in front of the webcam (this is the TARGET).
3. Toggle Face Swap **ON**.
4. Observe the live preview: Person A's face will appear synthetically plastered onto Person B's body.
5. Observe the result: If the attack is effective, Model A may ACCEPT the face. If C-Adv is robust against this specific attack, it may classify the frame as Impersonation, resulting in a final state of `SUSPECTED_IMPERSONATION`.
