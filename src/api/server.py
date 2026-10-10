import asyncio
import os
import cv2
import numpy as np
import glob
import base64
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.runtime.live_pipeline_v4 import LiveInferencePipelineV4

# Global state
pipeline_instance = None
pipeline_lock = asyncio.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_instance
    print("Starting API Server and initializing ML pipeline (V4)...")
    
    # Initialize the heavy ML pipeline strictly once
    pipeline_instance = LiveInferencePipelineV4(gallery_dir="experiments/live_demo/v4_gallery")
    
    # Configuration for the live demo reference identity
    demo_ref_dir = os.getenv("DEMO_REFERENCE_DIR", "sample_data/lfw/Angelina_Jolie")
    
    # Preserve existing initialization requirement for "Alice" identity
    alice_images = glob.glob(f"{demo_ref_dir}/*.jpg")[:5]
    if alice_images:
        pipeline_instance.identity_pipeline.enroll("Alice", alice_images)
        # Set alice as the initial source face
        alice_img = cv2.imread(alice_images[0])
        if alice_img is not None:
            faces = pipeline_instance.identity_pipeline.engine.extract_faces(alice_img)
            if faces:
                best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                pipeline_instance.swapper.set_source_face(best_face)
        print(f"Enrolled Alice using {len(alice_images)} images from {demo_ref_dir}.")
    else:
        print(f"Warning: Reference images not found in {demo_ref_dir}.")
        
    yield
    
    # Cleanup on shutdown
    pipeline_instance = None
    print("Shutting down API Server.")

app = FastAPI(lifespan=lifespan)

# Restrict CORS to expected Vite local development environment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {
        "status": "alive",
        "pipeline_initialized": pipeline_instance is not None
    }

@app.post("/enroll")
async def enroll_identity(identity_name: str = Form(...), files: List[UploadFile] = File(...)):
    if pipeline_instance is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
    if not files:
        raise HTTPException(status_code=400, detail="No reference images provided.")
        
    # Write uploaded files to temp directory
    temp_dir = f"experiments/live_demo/temp_enrollment/{identity_name}"
    os.makedirs(temp_dir, exist_ok=True)
    
    image_paths = []
    try:
        for file in files:
            content = await file.read()
            path = os.path.join(temp_dir, file.filename)
            with open(path, "wb") as f:
                f.write(content)
            image_paths.append(path)
            
        # Enroll via the existing identity_pipeline logic
        async with pipeline_lock:
            success = pipeline_instance.identity_pipeline.enroll(identity_name, image_paths)
            if not success:
                raise HTTPException(status_code=400, detail="Failed to enroll identity (no faces detected?)")
                
            # Grab one valid face to set as Native InSwapper source
            for p in image_paths:
                img = cv2.imread(p)
                if img is not None:
                    faces = pipeline_instance.identity_pipeline.engine.extract_faces(img)
                    if faces:
                        best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                        pipeline_instance.swapper.set_source_face(best_face)
                        break
            
            return {"status": "success", "identity": identity_name, "references": len(image_paths)}
    finally:
        # Cleanup temp files
        for p in image_paths:
            if os.path.exists(p):
                os.remove(p)

@app.websocket("/ws/inference")
async def websocket_inference(websocket: WebSocket, condition: str = "genuine"):
    await websocket.accept()
    
    if pipeline_instance is None:
        await websocket.send_json({"error": "Pipeline not initialized"})
        await websocket.close()
        return

    frame_id = 0
    try:
        while True:
            # Receive binary JPEG bytes
            data = await websocket.receive_bytes()
            
            if not data:
                await websocket.send_json({"error": "Empty frame received", "frame_id": frame_id})
                continue
                
            # Decode JPEG into BGR frame for OpenCV/Pipeline
            np_arr = np.frombuffer(data, np.uint8)
            frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame_bgr is None:
                await websocket.send_json({"error": "Malformed JPEG", "frame_id": frame_id})
                continue
                
            # Serialize access to the single LiveInferencePipeline instance
            # This protects the temporal smoothing state and model execution from race conditions
            async with pipeline_lock:
                try:
                    result = pipeline_instance.process_frame(frame_bgr, condition=condition, frame_id=frame_id)
                    # Return the exact post-swap frame so the browser can display it
                    if condition == "impersonation" and getattr(pipeline_instance, "last_processed_frame", None) is not None:
                        _, buffer = cv2.imencode('.jpg', pipeline_instance.last_processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        result["processed_frame_b64"] = base64.b64encode(buffer).decode('utf-8')
                        
                    await websocket.send_json(result)
                except Exception as e:
                    # Handle gracefully without exposing internal python stack traces
                    await websocket.send_json({"error": str(e), "frame_id": frame_id})
                    
            frame_id += 1
            
    except WebSocketDisconnect:
        # Expected client disconnect
        pass
    except Exception as e:
        # Unexpected socket-level exception
        pass
