import yaml
import cv2
from src.identity.arcface import IdentityEngine
from src.identity.gallery import IdentityGallery
from src.identity.enrollment import EnrollmentManager

class IdentityPipeline:
    def __init__(self, config_path="configs/baseline.yaml", gallery_dir=None):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        identity_cfg = self.config.get("identity", {})
        model_name = identity_cfg.get("model_name", "buffalo_l")
        det_thresh = identity_cfg.get("det_thresh", 0.5)
        device = identity_cfg.get("device", "cuda")
        self.match_threshold = identity_cfg.get("match_threshold", None)

        if gallery_dir is None:
            gallery_dir = self.config.get("paths", {}).get("gallery_dir", "gallery")

        print("Initializing Identity Engine...")
        self.engine = IdentityEngine(model_name=model_name, det_thresh=det_thresh, device=device)
        self.gallery = IdentityGallery(gallery_dir=gallery_dir)
        self.enrollment = EnrollmentManager(self.engine, self.gallery)

    def set_threshold(self, threshold: float):
        """Allows overriding the threshold after calibration."""
        self.match_threshold = threshold

    def enroll(self, identity_id, image_paths):
        return self.enrollment.enroll(identity_id, image_paths)

    def identify(self, image_bgr):
        """
        End-to-end identification for a live frame.
        """
        if self.match_threshold is None:
            raise ValueError("match_threshold is null in config and has not been calibrated.")

        faces = self.engine.extract_faces(image_bgr)
        if not faces:
            return {"identity": None, "similarity": 0.0, "status": "NO_FACE"}

        # If multiple faces are present, select the largest one
        if len(faces) > 1:
            best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            status_prefix = "MULTI_FACE_"
        else:
            best_face = faces[0]
            status_prefix = ""

        res = self.gallery.identify(self.engine, best_face.normed_embedding, threshold=self.match_threshold)
        
        # Prepend MULTI_FACE_ warning if multiple faces were detected but we processed one
        if len(faces) > 1:
            res["status"] = status_prefix + res["status"]
            
        return res
