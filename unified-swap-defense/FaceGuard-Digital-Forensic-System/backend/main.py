import os
import cv2
import numpy as np
import base64
import asyncio
import tempfile
import json
from collections import deque
import logging
import time
import uuid
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from model import load_faceguard_model, preprocess_frame

logger = logging.getLogger("faceguard")
app = FastAPI(title="FaceGuard API", version="3.0")

# Only the local dev UI may call the API from a browser (override with ALLOWED_ORIGINS="a,b").
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3005,http://127.0.0.1:3005,http://localhost:3000,http://127.0.0.1:3000").split(",")
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_VIDEO_BYTES = 200 * 1024 * 1024
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "faceguard_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
UPLOADS = {}  # opaque upload id -> file path inside UPLOAD_DIR (clients never choose server paths)


async def read_limited(file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload in chunks and reject anything over max_bytes (413) instead of buffering it all."""
    data = bytearray()
    while chunk := await file.read(1024 * 1024):
        data += chunk
        if len(data) > max_bytes:
            raise HTTPException(status_code=413, detail=f"File too large (max {max_bytes // (1024 * 1024)} MB)")
    return bytes(data)


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

@app.on_event("startup")
async def startup_event():
    global model
    model = load_faceguard_model()
    print(" FaceGuard model loaded.")


@app.get("/")
async def root():
    return {"status": "FaceGuard API is live", "version": "3.0"}


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


#  IMAGE ANALYSIS 
@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        contents = await read_limited(file, MAX_IMAGE_BYTES)
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file"})

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor  = preprocess_frame(img_rgb)
        score   = float(model.predict(tensor, verbose=0)[0][0])

        return {
            "score":      score,
            "verdict":    "FAKE" if score > 0.40 else "REAL",
            "confidence": score if score > 0.40 else 1 - score,
        }
    except HTTPException:
        raise
    except cv2.error:  # oversized / corrupt media: client error, and never echo OpenCV internals
        return JSONResponse(status_code=400, content={"error": "Invalid or too large media file"})
    except Exception:
        logger.exception("request failed")
        return JSONResponse(status_code=500, content={"error": "Internal error"})


#  VIDEO FORENSICS  SSE STREAMING 
@app.get("/analyze/video/stream")
async def analyze_video_stream(path: str):
    """
    SSE endpoint. Caller uploads video first via /upload/video,
    gets back a tmp path, then connects here to stream results.
    Sends JSON events: frame updates + final summary.
    """
    path = UPLOADS.pop(path, None)  # `path` is the opaque id from /upload/video, never a client-chosen file path
    if path is None or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Unknown or expired upload id")

    async def event_generator():
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            yield f"data: {json.dumps({'error': 'Cannot open video'})}\n\n"
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = 30

        score_buffer       = deque(maxlen=5)
        smoothed_predictions = []
        suspicious_frames  = []
        frame_id           = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_id % frame_interval == 0:
                    rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    def predict_frame(f):
                        return float(model.predict(preprocess_frame(f), verbose=0)[0][0])
                        
                    raw_score  = await asyncio.to_thread(predict_frame, rgb)
                    score_buffer.append(raw_score)
                    smoothed   = float(sum(score_buffer) / len(score_buffer))
                    smoothed_predictions.append(smoothed)
                    timestamp  = round(frame_id / fps, 2)
                    progress   = min(int(frame_id / max(total_frames, 1) * 100), 99)

                    # Encode current frame as base64 JPEG for display
                    _, buf = cv2.imencode(
                        ".jpg",
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 80]
                    )
                    frame_b64 = base64.b64encode(buf).decode("utf-8")

                    # Collect suspicious
                    if raw_score > 0.40:
                        small = cv2.resize(rgb, (160, 120))
                        _, sbuf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
                        suspicious_frames.append({
                            "time":      timestamp,
                            "score":     raw_score,
                            "thumbnail": f"data:image/jpeg;base64,{base64.b64encode(sbuf).decode()}"
                        })

                    event = {
                        "type":       "frame",
                        "frame_id":   frame_id,
                        "timestamp":  timestamp,
                        "raw_score":  raw_score,
                        "smoothed":   smoothed,
                        "progress":   progress,
                        "frame_b64":  frame_b64,
                    }
                    yield f"data: {json.dumps(event)}\n\n"

                    # Yield control so the event loop can send the SSE chunk
                    await asyncio.sleep(0)

                frame_id += 1

        finally:
            cap.release()

        # Final summary event
        scores    = smoothed_predictions
        avg_score = float(np.mean(scores)) if scores else 0.0
        timeline  = [
            {"time": round(i * frame_interval / fps, 2), "score": s}
            for i, s in enumerate(scores)
        ]

        summary = {
            "type":                   "complete",
            "verdict":                "FAKE" if avg_score > 0.40 else "REAL",
            "average_score":          avg_score,
            "timeline":               timeline,
            "suspicious_frames":      suspicious_frames[:8],
            "total_frames_analyzed":  len(scores),
            "progress":               100,
        }
        yield f"data: {json.dumps(summary)}\n\n"

        # Clean up temp file
        try:
            os.unlink(path)
        except Exception:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/upload/video")
async def upload_video(file: UploadFile = File(...)):
    """Upload video into the private upload dir; returns an opaque id for the SSE stream endpoint."""
    now = time.time()  # drop uploads nobody streamed within an hour (otherwise they accumulate on disk)
    for uid, path in list(UPLOADS.items()):
        if not os.path.exists(path) or now - os.path.getmtime(path) > 3600:
            UPLOADS.pop(uid, None)
            try: os.unlink(path)
            except OSError: pass
    contents = await read_limited(file, MAX_VIDEO_BYTES)
    upload_id = uuid.uuid4().hex
    tmp_path = os.path.join(UPLOAD_DIR, upload_id + ".mp4")
    with open(tmp_path, "wb") as f:
        f.write(contents)
    UPLOADS[upload_id] = tmp_path
    return {"path": upload_id, "size": len(contents)}


#  LEGACY: non-streaming video endpoint (kept for compatibility) 
@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    tmp_path = None
    try:
        contents = await read_limited(file, MAX_VIDEO_BYTES)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        cap          = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = 30
        score_buffer = deque(maxlen=5)
        smoothed_predictions = []; suspicious_frames = []; frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_id % frame_interval == 0:
                rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                raw       = float(model.predict(preprocess_frame(rgb), verbose=0)[0][0])
                score_buffer.append(raw)
                smoothed  = float(sum(score_buffer) / len(score_buffer))
                smoothed_predictions.append(smoothed)
                timestamp = round(frame_id / fps, 2)
                if raw > 0.40:
                    small = cv2.resize(rgb, (160, 120))
                    _, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
                    suspicious_frames.append({"time": timestamp, "score": raw, "thumbnail": f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"})
            frame_id += 1

        cap.release()
        scores    = smoothed_predictions
        avg_score = float(np.mean(scores)) if scores else 0.0
        return {
            "verdict": "FAKE" if avg_score > 0.40 else "REAL",
            "average_score": avg_score,
            "timeline": [{"time": round(i * frame_interval / fps, 2), "score": s} for i, s in enumerate(scores)],
            "suspicious_frames": suspicious_frames[:8],
            "total_frames_analyzed": len(scores),
        }
    except HTTPException:
        raise
    except cv2.error:  # oversized / corrupt media: client error, and never echo OpenCV internals
        return JSONResponse(status_code=400, content={"error": "Invalid or too large media file"})
    except Exception:
        logger.exception("request failed")
        return JSONResponse(status_code=500, content={"error": "Internal error"})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


#  LIVE WEBCAM WEBSOCKET 
@app.websocket("/ws/webcam")
async def webcam_ws(websocket: WebSocket):
    await websocket.accept()
    score_buffer = deque(maxlen=5)
    try:
        while True:
            data = await websocket.receive_text()
            if len(data) > MAX_IMAGE_BYTES * 2:  # base64 overhead margin
                await websocket.close(code=1009)
                break
            if "," in data:
                data = data.split(",")[1]
            img_bytes = base64.b64decode(data)
            nparr     = np.frombuffer(img_bytes, np.uint8)
            frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_json({"error": "bad frame"}); continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            def predict_frame(f):
                return float(model.predict(preprocess_frame(f), verbose=0)[0][0])
                
            raw = await asyncio.to_thread(predict_frame, rgb)
            score_buffer.append(raw)
            smoothed = float(sum(score_buffer) / len(score_buffer))

            await websocket.send_json({
                "raw_score":     raw,
                "smoothed_score": smoothed,
                "verdict":       "FAKE" if smoothed > 0.40 else "REAL",
            })
    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()
