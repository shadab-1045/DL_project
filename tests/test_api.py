import os
from fastapi.testclient import TestClient
from src.api.server import app

def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"
        assert response.json()["pipeline_initialized"] is True

def test_websocket_malformed():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/inference") as websocket:
            websocket.send_bytes(b"not a jpeg")
            data = websocket.receive_json()
            assert "error" in data
            assert data["error"] == "Malformed JPEG"

def test_websocket_empty():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/inference") as websocket:
            websocket.send_bytes(b"")
            data = websocket.receive_json()
            assert "error" in data
            assert data["error"] == "Empty frame received"

def test_websocket_valid_image():
    # Use an existing static image from the sample dataset
    img_path = "sample_data/lfw/Angelina_Jolie/00.jpg"
    assert os.path.exists(img_path)
    
    with open(img_path, "rb") as f:
        img_bytes = f.read()
        
    with TestClient(app) as client:
        with client.websocket_connect("/ws/inference?condition=genuine") as websocket:
            websocket.send_bytes(img_bytes)
            data = websocket.receive_json()
            
            # Verify the image successfully reached the pipeline boundary and returned JSON
            assert "timestamp" in data
            assert "frame_id" in data
            assert data["frame_id"] == 0
            assert "face_detected" in data
            # Since it's Angelina Jolie and Alice is enrolled with her, Model A should accept
            if data["face_detected"]:
                assert "raw_model_a_similarity" in data
