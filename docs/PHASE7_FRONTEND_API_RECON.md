# Phase 7A: Frontend & API Read-Only Reconnaissance

## 1. Current Repository State
A complete inspection of the repository directory structure confirms the following:
- **Backend/API**: No existing API code, web servers, or routing files exist.
- **Frontend/UI**: No existing frontend code, `package.json`, HTML, or CSS files exist.
- **Runtime Entry Points**: `src/runtime/manual_live_validation.py` currently serves as the terminal-based interactive entry point.
- **Test Infrastructure**: Pytest is configured in `tests/`, but there are no API tests.

## 2. Live Pipeline Interface
The core engine (`src/runtime/live_pipeline.py -> LiveInferencePipeline`) provides an exceptionally clean, decoupled programmatic interface:
- **Initialization**: `LiveInferencePipeline()` handles all model loading (Model A, InSwapper, C-Adv) natively in its constructor.
- **Webcam Ownership**: The pipeline **does not** own the webcam. It relies on the caller (`manual_live_validation.py`) to supply `frame_bgr` frames. This makes it instantly compatible with an API.
- **Input Format**: A raw numpy BGR image array (e.g., from `cv2.imdecode`).
- **Output Format**: The `process_frame` method returns a comprehensive dictionary (`log_record`) containing all necessary metrics, similarities, and states.
- **State Maintenance**: The pipeline maintains temporal state (`self.ema_similarity`, `self.ema_probs`) internally between frame calls.

## 3. Web / API Feasibility
- **Current Dependencies**: `requirements.txt` contains deep learning packages (PyTorch, InsightFace, ONNXRuntime, OpenCV). It **does not** currently contain `fastapi`, `uvicorn`, or `websockets`.
- **Minimum Backend Architecture**: 
  - A FastAPI application instantiated with a global `LiveInferencePipeline` loaded during the app's `lifespan` startup.
  - Endpoint A: `GET /health` (status check).
  - Endpoint B: `WebSocket /ws/inference` (bidirectional stream for receiving JPEG bytes and returning JSON results).

## 4. Model Lifecycle Risk
**Critical Risk**: `LiveInferencePipeline.__init__()` takes significant time and RAM/VRAM to load four heavy ONNX models and a PyTorch checkpoint. 
**Requirement**: The API **must** initialize the pipeline strictly once at server startup as a singleton, and persist it across all WebSocket connections. Instantiating the pipeline per HTTP request or per frame will cause immediate hardware exhaustion and catastrophic failure.

## 5. Browser / Video Architecture Options
**Option A (Recommended): Browser captures webcam → WebSocket → FastAPI inference**
- *Description*: A React web app accesses the user's camera, draws frames to a hidden canvas, encodes them as JPEG, and streams them over a WebSocket. FastAPI decodes the JPEG to BGR, calls `pipeline.process_frame()`, and returns the JSON result over the same WebSocket. The React app overlays the JSON data on the live video.
- *Pros*: Highly scalable, decoupled, standard modern web architecture. The backend can theoretically be hosted on a dedicated GPU server while the UI runs locally in any browser.
- *Cons*: Slight encoding/decoding overhead (negligible compared to model latency).

**Option B: Python owns webcam → HTTP MJPEG stream**
- *Description*: Python uses `cv2.VideoCapture(0)`, processes frames, draws text overlays, and streams an MJPEG feed to a simple HTML `<img>` tag.
- *Pros*: Extremely simple to implement.
- *Cons*: Forces the server and the camera to be on the exact same physical machine. Restricts deployment options. Not a standard production architecture.

**Conclusion**: Option A is strongly recommended for a final-year academic project, as it demonstrates modern cloud-ready full-stack engineering.

## 6. Recommended Frontend Stack
Since no frontend exists, the minimum viable modern stack is:
- **Framework**: React + Vite (fast, standard, easy to package).
- **Styling**: Tailwind CSS (rapid UI prototyping without writing custom CSS files).
- **Communication**: Native browser `WebSocket` API.

## 7. Proposed API / WebSocket Contract
The backend will expect a binary WebSocket message (JPEG bytes) or a JSON payload containing base64 image data.
The backend will return a JSON payload identical to the existing `log_record`, stripped of server-internal metadata:

```json
{
  "timestamp": 1791043321.233,
  "frame_id": 21,
  "condition": "genuine",
  "face_detected": true,
  "raw_model_a_similarity": 0.796,
  "ema_model_a_similarity": 0.790,
  "model_a_threshold": 0.244,
  "model_a_raw_accept": true,
  "matched_identity": "Alice",
  "raw_c_probs": [0.460, 0.510, 0.029],
  "raw_c_class": 1,
  "smoothed_class": 1,
  "final_state": "UNKNOWN",
  "total_latency": 3.695,
  "fps": 0.27
}
```

## 8. Security / Privacy Boundary
- **Client (Browser)**: Owns the webcam stream. Displays results. Has no access to model weights, gallery reference images, file paths, or training data.
- **Server (FastAPI)**: Owns the models and the `sample_data`/`gallery` directories. Receives transient frames, processes them in memory, and returns numeric metrics. Frames are not persisted to disk unless explicitly requested via an API toggle for diagnostics.

## 9. Dependencies to Add
The following must be added to `requirements.txt` in Phase 7:
- `fastapi`
- `uvicorn`
- `websockets` (or `python-multipart` if HTTP is needed)

## 10. Next Implementation Steps
1. Install FastAPI and Uvicorn into the virtual environment.
2. Create `src/api/server.py` implementing the global pipeline singleton and the WebSocket endpoint.
3. Test the WebSocket API via a simple Python client before building the React frontend.

---

### Final Answers to Directives
- **Is FastAPI already available?** No.
- **Does React/Vite/etc. already exist?** No.
- **Recommended Architecture**: Option A (React Webcam -> WebSocket -> FastAPI singleton).
- **Next single implementation task**: Install `fastapi` and `uvicorn`, and create the initial `src/api/server.py` backbone.
