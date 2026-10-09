import os
import numpy as np

class IdentityGallery:
    def __init__(self, gallery_dir="gallery"):
        self.gallery_dir = gallery_dir
        self.identities = {}
        self.load_gallery()

    def load_gallery(self):
        """Loads all embeddings from the gallery directory."""
        if not os.path.exists(self.gallery_dir):
            os.makedirs(self.gallery_dir)
            return

        for identity_id in os.listdir(self.gallery_dir):
            identity_path = os.path.join(self.gallery_dir, identity_id)
            if os.path.isdir(identity_path):
                emb_path = os.path.join(identity_path, "embedding.npy")
                if os.path.exists(emb_path):
                    self.identities[identity_id] = np.load(emb_path)

    def save_identity(self, identity_id, embedding):
        """Saves a normalized embedding to the gallery."""
        identity_path = os.path.join(self.gallery_dir, identity_id)
        if not os.path.exists(identity_path):
            os.makedirs(identity_path)
            
        emb_path = os.path.join(identity_path, "embedding.npy")
        # Ensure it's normalized before saving
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        np.save(emb_path, embedding)
        self.identities[identity_id] = embedding

    def identify(self, engine, live_embedding, threshold=0.45):
        """
        Matches a live embedding against the gallery.
        Returns a dict with identity, similarity, and status.
        """
        if live_embedding is None:
            return {"identity": None, "similarity": 0.0, "status": "NO_FACE"}

        best_identity = None
        best_similarity = -1.0

        for identity_id, enrolled_emb in self.identities.items():
            sim = engine.compute_similarity(live_embedding, enrolled_emb)
            if sim > best_similarity:
                best_similarity = sim
                best_identity = identity_id

        if best_similarity >= threshold:
            return {
                "identity": best_identity,
                "similarity": float(best_similarity),
                "status": "VERIFIED"
            }
        else:
            return {
                "identity": None,
                "similarity": float(best_similarity) if best_similarity > -1.0 else 0.0,
                "status": "UNKNOWN"
            }
