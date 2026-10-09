import os
import cv2
import numpy as np
import pytest
from src.runtime.identity_pipeline import IdentityPipeline

@pytest.fixture
def pipeline(tmp_path):
    p = IdentityPipeline()
    p.config["paths"]["gallery_dir"] = str(tmp_path / "test_gallery")
    # Must set a dummy threshold to test since it's null in config
    p.set_threshold(0.50)
    
    # Enroll a dummy user to test "unknown"
    # Use randn to ensure random vector is centered at 0
    emb = np.random.randn(512).astype(np.float32)
    emb /= np.linalg.norm(emb)
    p.gallery.save_identity("KnownUser", emb)
    return p

def test_no_face(pipeline):
    # Blank image (no face)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    res = pipeline.identify(img)
    assert res["status"] == "NO_FACE"
    assert res["identity"] is None

def test_unreadable_image(pipeline):
    res = pipeline.identify(None)
    assert res["status"] == "NO_FACE"

def test_unknown_identity(pipeline):
    # Mocking a face extraction to return a different embedding
    class DummyFace:
        def __init__(self, emb):
            self.bbox = [0, 0, 100, 100]
            self.normed_embedding = emb

    # Use randn to ensure it is orthogonal to the enrolled user
    diff_emb = np.random.randn(512).astype(np.float32)
    diff_emb /= np.linalg.norm(diff_emb)
    
    pipeline.engine.extract_faces = lambda x: [DummyFace(diff_emb)]
    
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    res = pipeline.identify(img)
    
    assert res["status"] == "UNKNOWN"
    assert res["identity"] is None

def test_multiple_faces(pipeline):
    # Mock extracting 2 faces
    # Match the fixture embedding
    emb = pipeline.gallery.identities["KnownUser"]

    class DummyFace:
        def __init__(self, emb, bbox):
            self.bbox = bbox
            self.normed_embedding = emb

    # Face 1 is smaller, Face 2 is larger
    face1 = DummyFace(np.random.randn(512).astype(np.float32), [0, 0, 50, 50])
    face2 = DummyFace(emb, [0, 0, 100, 100]) # Matches KnownUser
    
    pipeline.engine.extract_faces = lambda x: [face1, face2]
    
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    res = pipeline.identify(img)
    
    assert res["status"] == "MULTI_FACE_VERIFIED"
    assert res["identity"] == "KnownUser"

