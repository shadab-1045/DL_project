# Phase 6A — Reconnaissance Report

## 1. CURRENT REPOSITORY STRUCTURE
The project is well-structured into functional Python packages under `src/`.
- `src/data/`: Data generation, manipulation, and PyTorch dataset definitions (`model_c_dataset.py`).
- `src/evaluation/`: Evaluation scripts (e.g., `run_final_evaluation.py`).
- `src/identity/`: Core biometrics (`arcface.py`, `enrollment.py`, `gallery.py`).
- `src/models/`: Neural architectures (`model_c.py`).
- `src/preprocessing/`: Image alignment and cropping (`alignment.py`).
- `src/runtime/`: The skeleton of the runtime identity pipeline (`identity_pipeline.py`).
- `src/training/`: Training and adversarial scripts (`train_model_c.py`, `adversarial.py`).
- `experiments/`: Result artifacts and saved checkpoints.
- `models/`: Location of the local `inswapper_128.onnx` file.

**No frontend or API backend folders (e.g., `api/` or `ui/`) currently exist in the repository.**

## 2. EXISTING REUSABLE COMPONENTS
- `src.identity.arcface.IdentityEngine`: Handles ArcFace initialization, face detection, embedding extraction, and multi-face resolution (selects the largest bounding box).
- `src.identity.gallery.IdentityGallery`: Manages identity storage and similarity matching.
- `src.preprocessing.alignment.align_face`: Performs standard ArcFace 112x112 alignment given facial landmarks (kps) and converts to RGB.
- `src.models.model_c.AntiImpersonationModel`: Architecture for C-Adv.
- `src.runtime.identity_pipeline.IdentityPipeline`: Current baseline inference pipeline that wraps ArcFace and handles basic state (e.g. `NO_FACE`).

*These components are production-ready and will be directly reused to build the live pipeline without rewriting core model logic.*

## 3. PHASE 5 ARTIFACT VERIFICATION
All Phase 5 artifacts remain intact and frozen. 
- **Model A Checkpoint (`buffalo_l/w600k_r50.onnx`)**: Validated matching SHA-256 (`4C06341C33C2CA1F86781DAB0E829F88AD5B64BE9FBA56E56BC9EBDEFC619E43`).
- **C-Control Checkpoint (`experiments/model_c_control/best_model.pt`)**: Validated matching SHA-256 (`1CF70BA964B5E60C6632AC8532A7BA8BCEC9F4509E425CE9CE6DF71E00C86E51`).
- **C-Adv Checkpoint (`experiments/model_c_adv/best_model.pt`)**: Validated matching SHA-256 (`A562487A5B3CC085E3CC1EED6FBEA93748376FB75BBD2FBDF3D0E871F966AFBD`).
- **Model A Threshold (`experiments/final_evaluation/model_a_threshold.json`)**: Validated as 0.244529.

*No overwrites, recalibrations, or tunings have occurred.*

## 4. DEEP-LIVE-CAM STATUS
The full graphical Deep-Live-Cam application is **NOT** installed in the environment. However, the core underlying face-swapping engine (`insightface.model_zoo.get_model('models/inswapper_128.onnx')`) is already installed, working, and was used successfully in Phase 1-5 (`generate_controlled_swaps.py`).

**Dependency Status:** 
- `onnxruntime-gpu`, `insightface`, `opencv-python`, and `torch` are natively available and highly compatible with the target `inswapper_128.onnx` model.
- Because the core face-swap model is already present and proven, we can securely implement the live face-swapping directly within our demonstration pipeline script. 
*Attempting to run the full external Deep-Live-Cam GUI application in parallel with our FastAPI inference backend is dangerous due to VRAM limitations on the RTX 3050 (6GB).*

## 5. DEPENDENCY COMPATIBILITY
The existing environment natively supports all underlying mechanics of Deep-Live-Cam.
- Python 3.10
- PyTorch with CUDA support.
- ONNX Runtime (using CUDAExecutionProvider).
- OpenCV (cv2) for frame manipulation.
No new ML dependencies are required. We only need standard web/API packages: `fastapi`, `uvicorn`, and `websockets` for the backend. Node.js and React (`npm`) will be needed for the frontend.

## 6. WEBCAM/CAMERA STATUS
I executed a Python test (`cv2.VideoCapture(0)`) in the current runtime environment. 
- **Result:** `Webcam opened: True`.
- **Status:** A physical webcam is natively accessible. We can perform a genuine live operational demonstration.

## 7. GPU/MEMORY STATUS
The target environment has an **NVIDIA RTX 3050 Laptop GPU (6 GB VRAM)**.
Memory budget estimation:
- **InsightFace/ArcFace (`buffalo_l`)**: ~1.5 GB
- **Inswapper Model (`inswapper_128.onnx`)**: ~1.0 GB
- **Anti-Impersonation C-Adv (EfficientNet-B0)**: ~0.1 GB
- **Total ML overhead:** ~2.6 GB

Because all components can fit inside the 6GB VRAM budget *if loaded in the same Python process*, we strongly recommend implementing the live face-swap and inference logic within a single FastAPI worker. Running a completely separate Deep-Live-Cam process would duplicate the `buffalo_l` overhead, risking Out-Of-Memory (OOM) errors.

## 8. LIVE FRAME COMPATIBILITY
- Webcam `cv2.VideoCapture` outputs raw `BGR` numpy arrays.
- The InsightFace swapper (`inswapper_128`) consumes and outputs `BGR` arrays.
- Our ArcFace `IdentityEngine` expects `BGR` arrays.
- Our C-Adv preprocessing (`src/preprocessing/alignment.py`) explicitly accepts `BGR`, converts to `RGB`, and applies PIL/torchvision transforms.
*Conclusion:* Frame formats are perfectly aligned. No intermediate video streams or disk writes are required; we will pass BGR numpy arrays directly in-memory from the webcam to the swap model, and then to the inference pipeline.

## 9. MODEL A + C-ADV LIVE INFERENCE PLAN
1. Read `BGR` frame from webcam.
2. (Optional Demo Intervention) If Face-Swap is toggled ON, pass frame through `inswapper_128` to produce an impersonated `BGR` frame.
3. Pass `BGR` frame to `IdentityEngine.extract_faces()`.
4. If no face is detected, yield `NO_FACE`.
5. Select the largest face (the "probe").
6. Extract `probe.normed_embedding`.
7. Compare embedding against enrolled gallery. If highest similarity < `0.244529`, yield `UNKNOWN`.
8. If similarity >= `0.244529` (Model A accepts):
9. Align and crop the face using `align_face(frame, probe.kps)` to get 112x112 RGB.
10. Apply identical torchvision transforms (Resize 224, ToTensor, Normalize ImageNet).
11. Run `C-Adv(face_tensor, claimed_embedding_tensor)`.
12. If C-Adv predicts `impersonation` (2), yield `SUSPECTED_IMPERSONATION`.
13. If C-Adv predicts `genuine` (0), yield `VERIFIED`.
14. If C-Adv predicts `different_person` (1), yield `UNKNOWN`.

## 10. TEMPORAL SMOOTHING DESIGN
Raw frame-by-frame inference will flicker due to subtle bounding-box noise, motion blur, and minor occlusion. 
**Strategy: Exponential Moving Average (EMA) / Probability Smoothing**
Instead of smoothing the discrete states (which can be erratic), we will smooth the continuous outputs before applying the decision thresholds.
- Maintain a rolling EMA (e.g., $\alpha = 0.3$) for the Model A Similarity Score.
- Maintain a rolling EMA for the Model C softmax probabilities (Genuine, Different, Impersonation).
- The decision engine (Section 9) will evaluate the *smoothed* similarity and *smoothed* probabilities.
- Configurable window parameter: `alpha` (weight of the newest frame).

## 11. FASTAPI/WEBSOCKET STATUS
- **Status:** Not currently implemented.
- **Design:** We will create `src/api/main.py`. The backend will expose:
  - `POST /enroll`: Accepts images and an identity name.
  - `GET /ws/stream`: A WebSocket endpoint. The client sends a command to start the stream. The server captures webcam frames, processes the inference loop, and pushes JSON state packets back to the client at 10-15 FPS.

## 12. REACT DASHBOARD STATUS
- **Status:** Not currently implemented.
- **Design:** We will create a minimalistic React SPA using Vite (`frontend/`).
- **Structure:**
  - **Live Video Feed:** An `<img>` tag receiving base64-encoded JPEG frames over the WebSocket.
  - **Control Panel:** Buttons to "Enroll Alice", "Start Camera", and a toggle switch for "Activate Deep-Live-Cam Impersonation".
  - **Status Panel:** Displays Identity Matched, Similarity Score, Anti-Impersonation Probabilities, and a large color-coded banner for the Final State (`VERIFIED` [Green], `UNKNOWN` [Yellow], `SUSPECTED_IMPERSONATION` [Red]).

## 13. LIVE DEMONSTRATION PROTOCOL
**Enrolled Identity:** A consenting participant ("Alice").
1. **Enrollment Phase:** Capture 5 reference images of Alice through the UI and submit to `/enroll`.
2. **Genuine Condition:** Alice sits in front of the webcam (Swap Toggled OFF). The system should display `VERIFIED` with high similarity.
3. **Unknown Condition:** "Bob" sits in front of the webcam (Swap Toggled OFF). The system should display `UNKNOWN` due to low Model A similarity.
4. **Impersonation Condition:** Bob sits in front of the webcam and clicks "Activate Deep-Live-Cam". The underlying `inswapper` will actively map Alice's face onto Bob's face in real-time. The system should display `SUSPECTED_IMPERSONATION`, successfully demonstrating that C-Adv catches the synthetic presentation despite Model A's high similarity score.

## 14. LOGGING DESIGN
The FastAPI server will write a structured JSONL log to `experiments/live_demo/demo_session.log`.
**Schema:**
`{"timestamp": "...", "frame_id": 1, "swap_active": false, "status": "VERIFIED", "detected_faces": 1, "identity": "Alice", "similarity": 0.85, "c_adv_probs": {"genuine": 0.9, "diff": 0.05, "imp": 0.05}}`
No biometric embeddings or actual images will be saved to disk during the live loop to ensure performance and privacy.

## 15. PHASE 6 ACCEPTANCE CRITERIA
- [ ] A FastAPI server manages the `IdentityPipeline` and `inswapper` in the same process without OOM errors.
- [ ] A React dashboard can initiate the camera, display the feed, and show live inference metrics.
- [ ] Live inference executes Model A, uses the frozen `0.244529` threshold, and evaluates the frozen C-Adv checkpoint using the official Phase 5 logic.
- [ ] Deep-Live-Cam face-swapping can be dynamically toggled ON/OFF affecting the live frames.
- [ ] State flickering is mitigated via probability-level temporal smoothing.
- [ ] The full Genuine $\rightarrow$ Unknown $\rightarrow$ Impersonation protocol can be demonstrated visually in real-time.

## 16. BLOCKERS/RISKS
- **Webcam Access:** While cv2 opened the camera in the background, browser-based webcam access via React requires HTTPS or `localhost` context. We will stream the *backend's* cv2 capture over WebSocket rather than using `navigator.mediaDevices.getUserMedia()` to ensure the backend has direct, uncompressed access to the frames and can natively run the inswapper module without complex WebRTC setups.

## 17. EXACT FILES TO CREATE/MODIFY
- `src/runtime/live_pipeline.py` (Create) - Integrates `inswapper`, `IdentityPipeline`, `C-Adv`, and Temporal Smoothing.
- `src/api/main.py` (Create) - FastAPI WebSocket server.
- `frontend/*` (Create) - Vite + React application.

## 18. PROPOSED PHASE 6B IMPLEMENTATION PLAN
1. **Core Runtime:** Write `src/runtime/live_pipeline.py`. Encapsulate the full inference logic, preprocessing, temporal smoothing, and the optional `inswapper` module.
2. **Backend API:** Write `src/api/main.py` to wrap the `live_pipeline` in a WebSocket and provide enrollment HTTP endpoints. Install FastAPI/Uvicorn.
3. **Frontend UI:** Initialize a basic React/Vite project in `frontend/`. Build the dashboard layout and WebSocket listener.
4. **Integration Test:** Perform a dry-run to ensure FPS is acceptable and memory remains stable on the RTX 3050.
