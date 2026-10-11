import sys
import os
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(25_000_000))  # decompression-bomb guard, must precede the first cv2 import
import torch  # noqa: F401  must load before onnxruntime so it finds torch's CUDA 12 / cuDNN 9 DLLs

# Paths setup
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DL_PROJECT_DIR = os.path.dirname(ROOT_DIR)   # this folder lives inside the DL_project repository root
FACEGUARD_DIR = os.path.join(ROOT_DIR, "FaceGuard-Digital-Forensic-System")
FACEGUARD_BACKEND_DIR = os.path.join(FACEGUARD_DIR, "backend")

sys.path.insert(0, DL_PROJECT_DIR)
sys.path.insert(0, FACEGUARD_BACKEND_DIR)

# Change working directory so the DL_project code base's relative paths work
os.chdir(DL_PROJECT_DIR)

# Provide FaceGuard with its model path relative to the new CWD or absolutely
os.environ["MODEL_PATH"] = os.path.join(FACEGUARD_DIR, "models", "faceguard_phase2_finetuned.h5")
# The verdict does not use Model C, so skip loading it. Set DL_LOAD_MODEL_C=1 to load it anyway.
os.environ.setdefault("DL_LOAD_MODEL_C", "0")
# The verdict does not use Model C, so skip loading it. Set DL_LOAD_MODEL_C=1 to load it anyway.
os.environ.setdefault("DL_LOAD_MODEL_C", "0")
# The verdict does not use Model C, so skip loading it. Set DL_LOAD_MODEL_C=1 to load it anyway.
os.environ.setdefault("DL_LOAD_MODEL_C", "0")

# Force TensorFlow to use CPU to prevent cuDNN DLL missing errors on Windows
# and to prevent it from stealing VRAM from PyTorch (RTX 4060 has 8GB VRAM)
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio
from collections import deque
import base64
import numpy as np
import cv2

# Import the existing FastAPI app instances
import main as fg_main
from src.api.server import app as dl_app, lifespan as dl_lifespan
import src.api.server as dl_server

sys.path.insert(0, ROOT_DIR)
from swapdet import SwapDetector, decide
MAX_FRAME_BYTES = 2 * 1024 * 1024
# Decision statistic = mean of the top-2 raw P(swap) over the last WINDOW frames; at/above SWAP_THRESHOLD blocks, even if ArcFace matched.
# Why a window: single webcam frames are noisy. On 6 real frames of the demo user a swap scored 0.97,0.94,0.96,0.93,0.51,0.35 while genuine
# frames scored 0.01-0.07, so a per-frame 0.8 (and then 0.5) threshold let swaps through in unlucky poses.
# Why 0.3: held-out test (n=263+263) at 0.3 = 96.2% swaps caught / 6.1% per-frame false alarms (94.7% / 4.2% at 0.5); on the demo user's
# frames genuine <= 0.12 and swapped >= 0.35. Chosen with the demo user's frames in view (n=6) -> treat as calibrated for this camera, not proven.
SWAP_THRESHOLD = 0.3
WINDOW = 8
swap_detector = None

@asynccontextmanager
async def unified_lifespan(app: FastAPI):
    print("--- Starting Unified API ---")
    print("1/3 Initializing FaceGuard models (used by the /faceguard video/image forensics endpoints)...")
    await fg_main.startup_event()
    
    print("2/3 Loading swap detector...")
    global swap_detector
    swap_detector = SwapDetector()
    print("3/3 Initializing identity pipeline...")
    async with dl_lifespan(dl_app):
        print("--- Unified API Ready ---")
        yield

from fastapi.middleware.cors import CORSMiddleware
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

tracer_provider = TracerProvider()
tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

# Create the master API
app = FastAPI(
    title="Unified API (FaceGuard + identity pipeline)", 
    lifespan=unified_lifespan,
    telemetry={"tracer_provider": tracer_provider}
)

# Add CORS middleware to allow the React frontend to communicate with the root endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=fg_main.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OriginGuard:
    """Reject browser requests from foreign origins that change state or open a WebSocket. CORS only hides *responses*; a
    hostile page could still POST /dl/enroll (persisting an attacker's face) or open the WebSocket. Clients that send no
    Origin header (curl, tests, server-side tools) are unaffected."""
    def __init__(self, app, allowed):
        self.app, self.allowed = app, set(allowed)

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            origin = dict(scope["headers"]).get(b"origin")
            unsafe = scope["type"] == "websocket" or scope["method"] not in ("GET", "HEAD", "OPTIONS")
            if origin and unsafe and origin.decode() not in self.allowed:
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                else:
                    await JSONResponse({"error": "origin not allowed"}, status_code=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)

app.add_middleware(OriginGuard, allowed=fg_main.ALLOWED_ORIGINS)

# Mount the sub-applications
app.mount("/faceguard", fg_main.app)
app.mount("/dl", dl_app)

@app.websocket("/ws/unified-verify")
async def unified_verify_ws(websocket: WebSocket, condition: str = "genuine"):
    """
    Per frame: ArcFace identity (+ optional InSwapper attack when condition=impersonation), then the
    swap detector, which can veto an identity match.
    """
    await websocket.accept()
    
    if dl_server.pipeline_instance is None or fg_main.model is None:
        await websocket.send_json({"error": "Models not initialized"})
        await websocket.close()
        return

    if condition == "impersonation" and dl_server.pipeline_instance.swapper.source_face is None:
        await websocket.send_json({"error": "Attack simulation needs a source identity to impersonate. Run `python make_demo_identity.py` (or enroll one in the Deepfake Generator tab), then restart."})
        await websocket.close()
        return

    frame_id = 0
    window = deque(maxlen=WINDOW)
    try:
        while True:
            # Receive binary JPEG bytes (Optimization from previous step)
            data = await websocket.receive_bytes()
            if not data or len(data) > MAX_FRAME_BYTES: continue  # drop empty/oversized frames
            
            # Decode frame
            np_arr = np.frombuffer(data, np.uint8)
            frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame_bgr is None: continue
            
            # --- 1. Identity (ArcFace) + optional simulated swap attack (InSwapper) ---
            async with dl_server.pipeline_lock:
                dl_result = await asyncio.to_thread(
                    dl_server.pipeline_instance.process_frame, 
                    frame_bgr, condition, frame_id
                )
                scored_bgr = dl_server.pipeline_instance.last_processed_frame
                bbox = dl_server.pipeline_instance.last_face_bbox
            
            # --- 2. Swap detector on the face crop of the frame the verifier actually sees ---
            if bbox is None:
                window.clear()  # no face: reset temporal state
                p_swap = None
            else:
                window.append(await asyncio.to_thread(swap_detector.p_swap, scored_bgr, bbox))
                top = sorted(window)[-2:]
                p_swap = sum(top) / len(top)
            
            # --- 3. Decision: the detector can veto an identity match ---
            identity = dl_result.get("matched_identity")
            id_ok = bool(dl_result.get("model_a_accept", dl_result.get("model_a_raw_accept")))
            verdict, reason = decide(p_swap, id_ok, identity, SWAP_THRESHOLD, n_frames=len(window))
            
            processed_b64 = None
            if condition == "impersonation":  # show the user the frame the verifier actually sees (the swapped face)
                ok_enc, enc = cv2.imencode(".jpg", scored_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok_enc: processed_b64 = base64.b64encode(enc.tobytes()).decode()
            
            response = {
                "frame_id": frame_id,
                "processed_frame": processed_b64,
                "face_detected": bool(dl_result.get("face_detected")),
                "identity_threshold": dl_server.pipeline_instance.identity_pipeline.match_threshold,
                "swap_threshold": SWAP_THRESHOLD,
                "latency_ms": round(dl_result.get("latency_total", dl_result.get("total_latency", 0.0)) * 1000),
                "condition": condition,
                "swap_probability": p_swap,
                "identity": identity if id_ok else None,
                "identity_similarity": dl_result.get("model_a_similarity", dl_result.get("raw_model_a_similarity")),
                "identity_accepted_by_arcface": id_ok,
                "swap_applied": dl_result.get("swap_applied"),
                "overall_verdict": verdict,
                "reason": reason,
                "dl_project_result": dl_result,
            }
            
            await websocket.send_json(response)
            frame_id += 1
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:  # fail closed and tell the UI, instead of silently freezing the last verdict
            await websocket.send_json({"error": f"Verification error: {type(e).__name__}"})
        except Exception:
            pass

@app.get("/")
def root():
    return {
        "message": "Unified API is running",
        "endpoints": {
            "faceguard": "/faceguard",
            "dl_project": "/dl"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "alive", "message": "Unified API is healthy"}

if __name__ == "__main__":
    print("Starting server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
