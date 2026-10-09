import cv2
import numpy as np
from insightface.app import FaceAnalysis

class IdentityEngine:
    def __init__(self, model_name="buffalo_l", det_thresh=0.5, device="cuda"):
        # Initialize FaceAnalysis pipeline
        providers = ['CUDAExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=0 if device == "cuda" else -1, det_thresh=det_thresh)

    def extract_faces(self, image_bgr):
        """
        Detects faces, aligns them, and extracts features.
        Returns a list of Face objects (from insightface).
        """
        if image_bgr is None:
            return []
        faces = self.app.get(image_bgr)
        return faces

    def get_embedding(self, face_image):
        """
        Returns the normalized embedding of the most prominent face in the image.
        If no face is detected, returns None.
        """
        faces = self.extract_faces(face_image)
        if not faces:
            return None
        
        # Select the largest face by bounding box area
        best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return best_face.normed_embedding

    def compute_similarity(self, emb1, emb2):
        """
        Computes cosine similarity between two normalized embeddings.
        """
        if emb1 is None or emb2 is None:
            return 0.0
        return np.dot(emb1, emb2)
