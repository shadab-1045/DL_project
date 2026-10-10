import os
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from src.runtime.identity_pipeline import IdentityPipeline
from src.runtime.live_swap import NativeInSwapper
from src.models.model_c_v4 import ModelCV4
from src.preprocessing.alignment import align_face

class LiveInferencePipelineV4:
    def __init__(self, 
                 fusion_checkpoint="experiments/model_c_v4/best_fusion_model.pt", 
                 model_a_threshold=0.244529, device="cuda", ema_alpha=0.3,
                 gallery_dir="experiments/live_demo/v4_gallery"):
        self.device = device
        self.ema_alpha = ema_alpha
        
        # 1. Initialize Identity Engine (Model A)
        print("Initializing Identity Engine (Model A)...")
        self.identity_pipeline = IdentityPipeline(gallery_dir=gallery_dir)
        self.identity_pipeline.set_threshold(model_a_threshold)
        
        # 2. Initialize Native InSwapper
        print("Initializing Native InSwapper...")
        self.swapper = NativeInSwapper()
        
        # 3. Initialize Model C V4
        print("Initializing Model C V4...")
        self.c_adv_v4 = ModelCV4().to(device)
        if os.path.exists(fusion_checkpoint):
            # weights_only=False needed because of PyTorch serialization quirks with sklearn if present, but we just load state dict
            self.c_adv_v4.load_state_dict(torch.load(fusion_checkpoint, map_location=device))
        else:
            raise FileNotFoundError(f"V4 checkpoint not found: {fusion_checkpoint}")
            
        self.c_adv_v4.eval()
        
        # V4 Preprocessing
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
        os.makedirs("experiments/live_demo/v4", exist_ok=True)

    def start_session(self, session_id):
        self.session_id = session_id
        self.log_file = open(f"experiments/live_demo/v4/{session_id}.jsonl", "a")
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
        
        # Step 1: Optional Swap (native InSwapper-based live face-swap presentation attack)
        swap_start = time.time()
        if condition == "impersonation":
            target_faces = self.identity_pipeline.engine.extract_faces(frame_bgr)
            processed_frame = self.swapper.process_frame(frame_bgr, target_faces)
            swap_applied = True
        else:
            processed_frame = frame_bgr
            swap_applied = False
        swap_latency = time.time() - swap_start
        
        self.last_processed_frame = processed_frame
        
        # Step 2: Live Model A Inference & Face Detection
        detect_start = time.time()
        faces = self.identity_pipeline.engine.extract_faces(processed_frame)
        detect_latency = time.time() - detect_start
        
        if not faces:
            return self._log_and_return_empty("NO_FACE", start_time, condition, frame_id, swap_applied, swap_latency, detect_latency)
            
        model_a_start = time.time()
        best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = best_face.normed_embedding
        
        id_res = self.identity_pipeline.gallery.identify(
            self.identity_pipeline.engine, emb, threshold=self.identity_pipeline.match_threshold
        )
        model_a_latency = time.time() - model_a_start
        
        similarity = id_res["similarity"]
        matched_identity = id_res["identity"]
        
        if not self.initialized_ema:
            self.ema_similarity = similarity
        else:
            self.ema_similarity = self.ema_alpha * similarity + (1 - self.ema_alpha) * self.ema_similarity

        raw_accepted = similarity >= self.identity_pipeline.match_threshold
        
        # Step 3: Model C V4 Inference
        visual_latency = 0.0
        fusion_latency = 0.0
        raw_probs = [0.0, 0.0, 0.0]
        raw_class = -1
        p_synth_val = 0.0
        p_id_val = 0.0
        
        if raw_accepted and matched_identity is not None:
            c_start = time.time()
            
            # Align face
            aligned_rgb = align_face(processed_frame, best_face.kps)
            from src.preprocessing.v4_live_parity import V4LiveParityAdapter
            parity_adapter = V4LiveParityAdapter()
            face_tensor = parity_adapter.preprocess(aligned_rgb).to(self.device)
            sim_tensor = torch.tensor([[similarity]], dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                # Extract branch probabilities
                vis_out = self.c_adv_v4.visual_branch(face_tensor)
                visual_latency = time.time() - c_start
                
                f_start = time.time()
                logits, p_synth, p_id = self.c_adv_v4(face_tensor, sim_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                fusion_latency = time.time() - f_start
                
                p_synth_val = p_synth.item()
                p_id_val = p_id.item()
                
            raw_probs = probs.tolist()
            raw_class = int(np.argmax(probs))
            
            if not self.initialized_ema:
                self.ema_probs = probs
                self.initialized_ema = True
            else:
                self.ema_probs = self.ema_alpha * probs + (1 - self.ema_alpha) * self.ema_probs
                
        else:
            self.ema_probs = np.array([0.0, 1.0, 0.0]) # different_person
            self.initialized_ema = True

        # Step 4: Final UI Decision Logic (State Mapping)
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
            "swap_applied": swap_applied,
            "raw_model_a_similarity": float(similarity),
            "ema_model_a_similarity": float(self.ema_similarity),
            "model_a_threshold": float(self.identity_pipeline.match_threshold),
            "model_a_raw_accept": bool(raw_accepted),
            "matched_identity": matched_identity,
            "raw_c_class": raw_class,
            "raw_c_probs": [p_id_val, p_synth_val, 0.0],
            "smoothed_class": smoothed_class,
            "ema_c_probs": self.ema_probs.tolist(),
            "raw_predicted_state": raw_final_state,
            "final_state": final_state,
            "swap_latency": float(swap_latency),
            "model_a_latency": float(model_a_latency),
            "model_c_latency": float(visual_latency + fusion_latency),
            "total_latency": float(total_latency),
            "fps": float(fps),
            "status": "success",
            "error": None
        }
        
        if self.log_file:
            self.log_file.write(json.dumps(log_record) + "\\n")
            self.log_file.flush()
            
        return log_record

    def _log_and_return_empty(self, status, start_time, condition, frame_id, swap_applied, swap_latency, detect_latency):
        self.initialized_ema = False
        log_record = {
            "timestamp": time.time(),
            "session_id": self.session_id,
            "frame_id": frame_id,
            "condition": condition,
            "face_detected": False,
            "swap_applied": swap_applied,
            "raw_model_a_similarity": None,
            "ema_model_a_similarity": None,
            "model_a_threshold": float(self.identity_pipeline.match_threshold),
            "model_a_raw_accept": False,
            "matched_identity": None,
            "raw_c_class": None,
            "raw_c_probs": None,
            "smoothed_class": None,
            "ema_c_probs": None,
            "raw_predicted_state": "UNKNOWN",
            "final_state": "UNKNOWN",
            "swap_latency": float(swap_latency),
            "model_a_latency": None,
            "model_c_latency": None,
            "total_latency": time.time() - start_time,
            "fps": 0.0,
            "status": status,
            "error": status
        }
        if self.log_file:
            self.log_file.write(json.dumps(log_record) + "\\n")
            self.log_file.flush()
        return log_record
