# Phase 7B: FastAPI / WebSocket Backend Implementation

This document details the backend adapter implemented to bridge the frozen ML inference pipeline with the future React frontend.

## 1. Architectural Role
The FastAPI backend acts strictly as a web-socket adapter. It wraps the existing `LiveInferencePipeline` class in a persistent singleton instance and serializes access to it via a lock. 
**Crucially, it does NOT re-implement any Model A, C-Adv, temporal smoothing, or decision logic.**

## 2. Final Dependencies
The core reproducible Phase 5 dependencies are strictly pinned in `requirements.txt`:
- `torch==2.5.1+cu121`
- `torchvision==0.20.1+cu121`
- `insightface==0.7.3`
- `onnxruntime-gpu==1.23.2`
- `opencv-python==5.0.0.93`
- `numpy==1.26.4` (Note: Pinned to `1.26.4` rather than `2.2.6` because `insightface==0.7.3` binary wheels use the numpy 1.x C-API and will crash with `ValueError: numpy.dtype size changed` on numpy 2.x)

The following Phase 7 specific dependencies were explicitly added:
- `fastapi` (required for API routing and the lifespan API)
- `uvicorn` (required as the ASGI web server)
- `websockets` (required natively by both Uvicorn/FastAPI for WebSockets and by `smoke_test.py` as a client)
- `httpx` (required for automated Pytest `TestClient` tests)
- `requests` (required for synchronous `/health` endpoint checks in `smoke_test.py`)

## 3. Files Created
- `src/api/__init__.py`: Package init.
- `src/api/server.py`: The FastAPI application, defining the `lifespan`, `/health`, and `/ws/inference` routes.
- `tests/test_api.py`: Automated Pytest coverage for the API endpoints.
- `smoke_test.py`: A manual script to hit the live server endpoints.

## 4. API Endpoints

### `GET /health`
Returns a JSON object detailing whether the server is alive and whether the heavy ML models have finished loading into RAM/VRAM.
```json
{
  "status": "alive",
  "pipeline_initialized": true
}
```

### `WebSocket /ws/inference?condition=genuine`
Bidirectional stream for real-time inference.
- **Client → Server**: Binary JPEG bytes of the webcam frame.
- **Server → Client**: JSON inference state dumped exactly as returned by `LiveInferencePipeline.process_frame`.

## 5. Model Lifecycle & Explicit Reference Configuration
- **Lifecycle**: The `LiveInferencePipeline` takes ~45-60 seconds to load into memory. It is initialized exactly **once** inside the FastAPI `@asynccontextmanager def lifespan(app)` function. There is zero per-frame or per-request model loading.
- **Explicit Reference-Image Configuration**: Instead of hardcoding the identity path, the server uses an explicit environment variable `DEMO_REFERENCE_DIR` to determine the enrollment reference directory.
- **Default Demo Directory**: To ensure existing demonstration reproducibility, `DEMO_REFERENCE_DIR` gracefully defaults to `sample_data/lfw/Angelina_Jolie`. The system automatically grabs the first 5 images in this directory to enroll "Alice".
- **Concurrency**: Because the pipeline contains mutable temporal state (`self.ema_similarity`) and is inherently single-threaded on CPU, access to `pipeline.process_frame` inside the WebSocket loop is protected by an `asyncio.Lock()`. This prevents state corruption if multiple frames (or multiple clients) arrive simultaneously.

## 6. Error Handling
The WebSocket route was designed not to crash the server when given bad inputs. It catches:
- Empty frames -> Returns `{"error": "Empty frame received"}`.
- Malformed JPEG bytes -> Returns `{"error": "Malformed JPEG"}`.
- Hard Inference crashes -> Returns `{"error": "Inference exception occurred"}`.
Internal Python exception stack traces are swallowed and not forwarded to the client browser.

## 7. Testing Performed
### Unit Tests (`pytest tests/test_api.py`)
- Verified `/health` endpoint behavior.
- Verified WebSocket correctly handles `b"not a jpeg"` returning a JSON error.
- Verified WebSocket correctly handles `b""` returning an empty frame error.
- Verified WebSocket successfully decodes a real JPEG (`Angelina_Jolie/00.jpg`), feeds it to the pipeline boundary, and returns the expected structured JSON.

### Manual Smoke Test
- **Command used to start server**: `uvicorn src.api.server:app --port 8000`
- **Result**: The server initialized cleanly. `smoke_test.py` successfully connected via WebSocket, sent `00.jpg`, and received a `final_state: VERIFIED` JSON response. No exceptions occurred.

## 8. Known Limitations
- The backend relies entirely on the ML environment's execution provider. Without CUDA/TensorRT properly configured in the subprocess environment, the `total_latency` per frame remains high (several seconds).
- The `asyncio.Lock()` ensures thread safety but essentially restricts throughput to exactly one frame at a time globally. This is perfectly acceptable for the local Phase 7 demo scope.
- **Experimental Demo Identity**: The default reference identity ("Alice" enrolled from Angelina Jolie images) is explicitly an experimental demonstration identity. It does not represent a dynamic, user-authenticated account or database-backed enrollment system.
