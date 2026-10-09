import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

def verify_buffalo_l():
    print("Initializing FaceAnalysis with buffalo_l...")
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Model initialized successfully.")
    
    # Create a dummy image
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Run detection
    faces = app.get(img)
    print(f"Detected {len(faces)} faces in dummy image.")
    return True

if __name__ == '__main__':
    verify_buffalo_l()
