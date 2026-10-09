import os
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from src.runtime.identity_pipeline import IdentityPipeline
from src.runtime.live_swap import NativeInSwapper
from src.models.model_c import AntiImpersonationModel
from src.preprocessing.alignment import align_face

class LiveInferencePipeline:
    def __init__(self, c_adv_checkpoint="experiments/model_c_adv/best_model.pt", 
                 model_a_threshold=0.244529, device="cuda", ema_alpha=0.3,
                 gallery_dir="gallery"):
        self.device = device
        self.ema_alpha = ema_alpha
        
        # 1. Initialize Identity Engine (Model A)
        print("Initializing Identity Engine...")
        self.identity_pipeline = IdentityPipeline(gallery_dir=gallery_dir)
        self.identity_pipeline.set_threshold(model_a_threshold)
        
        # 2. Initialize Native InSwapper
        print("Initializing Native InSwapper...")
        self.swapper = NativeInSwapper()
        
        # 3. Initialize C-Adv
        print("Initializing C-Adv...")
        self.c_adv = AntiImpersonationModel()
        if os.path.exists(c_adv_checkpoint):
            self.c_adv.load_state_dict(torch.load(c_adv_checkpoint, map_location=device, weights_only=True))
        else:
            raise FileNotFoundError(f"C-Adv checkpoint not found: {c_adv_checkpoint}")
        self.c_adv.to(device)
        self.c_adv.eval()
        
        # C-Adv Preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Temporal State
        self.reset_temporal_state()
        
        # Logging setup
        self.session_id = None
        self.log_file = None

    def start_session(self, session_id):
        self.session_id = session_id
        os.makedirs("experiments/live_demo", exist_ok=True)
        self.log_file = open(f"experiments/live_demo/{session_id}.jsonl", "a")
        self.reset_temporal_state()
        
    def end_session(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None
            
    def reset_temporal_state(self):
        self.ema_similarity = 0.0
        self.ema_probs = np.array([1.0, 0.0, 0.0]) # Default genuine
        self.initialized_ema = False

    def process_frame(self, frame_bgr, condition="genuine", frame_id=0):
        start_time = time.time()
        
        # Step 1: Optional Swap (Impersonation Condition)
        swap_start = time.time()
        if condition == "impersonation":
            # Extract target faces
            target_faces = self.identity_pipeline.engine.extract_faces(frame_bgr)
            processed_frame = self.swapper.process_frame(frame_bgr, target_faces)
        else:
            processed_frame = frame_bgr
        swap_latency = time.time() - swap_start
        
        # Save post-swap frame for demo server display
        self.last_processed_frame = processed_frame
        
        # Step 2: Live Model A Inference
        model_a_start = time.time()
        faces = self.identity_pipeline.engine.extract_faces(processed_frame)
        if not faces:
            return self._log_and_return_empty("NO_FACE", start_time, condition, frame_id)
            
        best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = best_face.normed_embedding
        
        id_res = self.identity_pipeline.gallery.identify(
            self.identity_pipeline.engine, emb, threshold=self.identity_pipeline.match_threshold
        )
        model_a_latency = time.time() - model_a_start
        
        similarity = id_res["similarity"]
        matched_identity = id_res["identity"]
        
        # Temporal Smoothing for Similarity
        if not self.initialized_ema:
            self.ema_similarity = similarity
        else:
            self.ema_similarity = self.ema_alpha * similarity + (1 - self.ema_alpha) * self.ema_similarity

        raw_accepted = similarity >= self.identity_pipeline.match_threshold
        
        # Step 3: Model C-Adv Inference
        model_c_latency = 0.0
        raw_probs = [0.0, 0.0, 0.0]
        raw_class = -1
        
        if raw_accepted and matched_identity is not None:
            c_start = time.time()
            
            # Align face and preprocess
            aligned_rgb = align_face(processed_frame, best_face.kps)
            face_tensor = self.transform(aligned_rgb).unsqueeze(0).to(self.device)
            
            # Claimed identity embedding (from gallery, not the probe itself)
            claimed_emb = self.identity_pipeline.gallery.identities[matched_identity]
            claimed_emb_tensor = torch.tensor(claimed_emb, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.c_adv(face_tensor, claimed_emb_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                
            raw_probs = probs.tolist()
            raw_class = int(np.argmax(probs))
            
            # Temporal Smoothing for Probabilities
            if not self.initialized_ema:
                self.ema_probs = probs
                self.initialized_ema = True
            else:
                self.ema_probs = self.ema_alpha * probs + (1 - self.ema_alpha) * self.ema_probs
                
            model_c_latency = time.time() - c_start
            
        else:
            # Identity Rejected
            self.ema_probs = np.array([0.0, 1.0, 0.0]) # different_person
            self.initialized_ema = True

        # Step 4: Final Decision Logic
        raw_final_state = "UNKNOWN"
        if not raw_accepted:
            raw_final_state = "UNKNOWN"
        else:
            if raw_class == 0:
                raw_final_state = "VERIFIED"
            elif raw_class == 2:
                raw_final_state = "SUSPECTED_IMPERSONATION"
            else:
                raw_final_state = "UNKNOWN"

        # Smoothed state
        smoothed_class = int(np.argmax(self.ema_probs))
        if not raw_accepted:
            final_state = "UNKNOWN"
        else:
            if smoothed_class == 0:
                final_state = "VERIFIED"
            elif smoothed_class == 2:
                final_state = "SUSPECTED_IMPERSONATION"
            else:
                final_state = "UNKNOWN"
                
        total_latency = time.time() - start_time
        fps = 1.0 / total_latency if total_latency > 0 else 0.0
        
        log_record = {
            "timestamp": time.time(),
            "session_id": self.session_id,
            "frame_id": frame_id,
            "condition": condition,
            "face_detected": True,
            "raw_model_a_similarity": float(similarity),
            "ema_model_a_similarity": float(self.ema_similarity),
            "model_a_threshold": self.identity_pipeline.match_threshold,
            "model_a_raw_accept": bool(raw_accepted),
            "matched_identity": matched_identity,
            "raw_c_probs": raw_probs,
            "raw_c_class": raw_class,
            "raw_predicted_state": raw_final_state,
            "ema_c_probs": self.ema_probs.tolist(),
            "smoothed_class": smoothed_class,
            "final_state": final_state,
            "swap_latency": float(swap_latency),
            "model_a_latency": float(model_a_latency),
            "model_c_latency": float(model_c_latency),
            "total_latency": float(total_latency),
            "fps": float(fps)
        }
        
        if self.log_file:
            self.log_file.write(json.dumps(log_record) + "\n")
            self.log_file.flush()
            
        return log_record

    def _log_and_return_empty(self, status, start_time, condition, frame_id):
        self.initialized_ema = False
        log_record = {
            "timestamp": time.time(),
            "session_id": self.session_id,
            "frame_id": frame_id,
            "condition": condition,
            "face_detected": False,
            "status": status,
            "raw_predicted_state": "UNKNOWN",
            "model_a_raw_accept": False,
            "final_state": "UNKNOWN",
            "total_latency": time.time() - start_time
        }
        if self.log_file:
            self.log_file.write(json.dumps(log_record) + "\n")
            self.log_file.flush()
        return log_record
