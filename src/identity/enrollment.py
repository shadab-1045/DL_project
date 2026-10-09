import os
import cv2
import numpy as np
from src.identity.arcface import IdentityEngine
from src.identity.gallery import IdentityGallery

class EnrollmentManager:
    def __init__(self, engine: IdentityEngine, gallery: IdentityGallery):
        self.engine = engine
        self.gallery = gallery

    def enroll(self, identity_id, image_paths):
        """
        Enrolls an identity using multiple reference images.
        Extracts embeddings, averages them, normalizes the result, and saves it.
        """
        embeddings = []
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            emb = self.engine.get_embedding(img)
            if emb is not None:
                embeddings.append(emb)
            else:
                print(f"Warning: No face detected in {img_path}")

        if not embeddings:
            print(f"Error: Could not extract any valid face embeddings for {identity_id}")
            return False

        # Aggregate embeddings (mean pooling)
        aggregated_emb = np.mean(embeddings, axis=0)
        
        # Save to gallery (it will normalize inside)
        self.gallery.save_identity(identity_id, aggregated_emb)
        print(f"Successfully enrolled {identity_id} using {len(embeddings)} reference images.")
        return True
