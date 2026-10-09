import asyncio
import os
import glob
import cv2
import json
import time
import requests
import websockets

async def test_integration():
    print("--- API Server Integration Test ---")
    
    # 1. Wait for Server Health
    print("Waiting for server to be healthy...")
    healthy = False
    for i in range(30):
        try:
            r = requests.get("http://127.0.0.1:8000/health")
            if r.status_code == 200 and r.json().get("pipeline_initialized") == True:
                healthy = True
                print("Server is healthy!")
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
        
    if not healthy:
        print("Server failed to become healthy within 30s")
        return
        
    # 2. Test Enrollment
    print("Testing Enrollment...")
    alice_images = glob.glob("sample_data/lfw/Angelina_Jolie/*.jpg")[:3]
    files = []
    for p in alice_images:
        files.append(("files", (os.path.basename(p), open(p, "rb"), "image/jpeg")))
        
    r = requests.post("http://127.0.0.1:8000/enroll", data={"identity_name": "TestAlice"}, files=files)
    print("Enroll Status:", r.status_code)
    print("Enroll Output:", r.json())
    
    # 3. Test WebSocket Inference
    print("Testing WebSocket Inference...")
    ws_url = "ws://127.0.0.1:8000/ws/inference?condition=genuine"
    try:
        async with websockets.connect(ws_url) as ws:
            print("WebSocket Connected!")
            
            # Send genuine frame
            test_frame_path = "sample_data/lfw/Angelina_Jolie/Angelina_Jolie_0006.jpg" # Actually "06.jpg" in previous globs, let's just grab the 4th image
            test_frame_path = glob.glob("sample_data/lfw/Angelina_Jolie/*.jpg")[3]
            
            # Read and encode to JPEG exactly like browser does
            img = cv2.imread(test_frame_path)
            _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            
            print(f"Sending {len(buffer.tobytes())} bytes...")
            await ws.send(buffer.tobytes())
            
            # Wait for response
            response = await ws.recv()
            data = json.loads(response)
            
            print("\n--- INFERENCE RESULT ---")
            print(f"Face Detected: {data.get('face_detected')}")
            print(f"Model A Similarity: {data.get('model_a_similarity')}")
            print(f"Model A Accept: {data.get('model_a_accept')}")
            print(f"Final UI State: {data.get('final_ui_state')}")
            print("------------------------")
            
    except Exception as e:
        print("WebSocket Test Failed:", e)

if __name__ == "__main__":
    asyncio.run(test_integration())
