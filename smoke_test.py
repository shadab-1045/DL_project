import time
import requests
import asyncio
import websockets
import json

def test_health():
    print("Testing /health...")
    resp = requests.get("http://localhost:8000/health")
    resp.raise_for_status()
    print("Health response:", resp.json())

async def test_websocket():
    print("Testing WebSocket...")
    uri = "ws://localhost:8000/ws/inference?condition=genuine"
    async with websockets.connect(uri) as ws:
        with open("sample_data/lfw/Angelina_Jolie/00.jpg", "rb") as f:
            img_bytes = f.read()
            
        print("Sending JPEG frame...")
        await ws.send(img_bytes)
        
        response = await ws.recv()
        data = json.loads(response)
        print("Received inference result keys:", list(data.keys()))
        print("Face detected:", data.get("face_detected"))
        print("Final state:", data.get("final_state"))

if __name__ == "__main__":
    test_health()
    asyncio.run(test_websocket())
    print("Smoke test completed successfully!")
