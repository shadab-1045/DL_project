import cv2
import json
import torch
import numpy as np
import time
from src.runtime.live_pipeline import LiveInferencePipeline
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Initializing pipeline...")
    pipeline = LiveInferencePipeline()
    
    # We need the Alice enrolled embedding since the test assumes Alice was enrolled
    # But wait, we can just enroll Alice again to populate the gallery
    import glob
    alice_images = glob.glob("sample_data/lfw/Angelina_Jolie/*.jpg")[:5]
    if not alice_images:
        print("Alice images missing.")
        return
    pipeline.identity_pipeline.enroll("Alice", alice_images)
    
    # Also Angelina_Jolie was matched. Let's see if we match Alice or Angelina_Jolie
    
    # Load diagnostic image
    img_path = "experiments/live_demo/manual/diagnostic_impersonation.jpg"
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Failed to load {img_path}")
        return
        
    print(f"Loaded {img_path}")
    
    # Run the exact inference logic (bypass webcam and native inswapper swap, just feed it as a regular frame)
    # The diagnostic frame already has the overlay text on it, but let's see what happens.
    
    start_time = time.time()
    
    # Extract faces
    faces = pipeline.identity_pipeline.engine.extract_faces(img_bgr)
    if not faces:
        print("No face detected in diagnostic image.")
        return
        
    best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = best_face.normed_embedding
    
    # Model A Identity Match
    id_res = pipeline.identity_pipeline.gallery.identify(
        pipeline.identity_pipeline.engine, emb, threshold=pipeline.identity_pipeline.match_threshold
    )
    
    similarity = id_res["similarity"]
    raw_accepted = similarity >= pipeline.identity_pipeline.match_threshold
    matched_identity = id_res["identity"]
    
    print(f"Model A Similarity: {similarity:.4f}")
    print(f"Model A Threshold: {pipeline.identity_pipeline.match_threshold}")
    print(f"Model A Accepted: {raw_accepted}")
    print(f"Matched Identity: {matched_identity}")
    
    # C-Adv inference
    if raw_accepted and matched_identity is not None:
        from src.preprocessing.alignment import align_face
        import torch.nn.functional as F
        
        aligned_rgb = align_face(img_bgr, best_face.kps)
        face_tensor = pipeline.transform(aligned_rgb).unsqueeze(0).to(pipeline.device)
        
        claimed_emb = pipeline.identity_pipeline.gallery.identities[matched_identity]
        claimed_emb_tensor = torch.tensor(claimed_emb, dtype=torch.float32).unsqueeze(0).to(pipeline.device)
        
        with torch.no_grad():
            logits = pipeline.c_adv(face_tensor, claimed_emb_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            
        raw_probs = probs.tolist()
        raw_class = int(np.argmax(probs))
        
        label_map = {0: "genuine", 1: "different_person", 2: "impersonation"}
        
        print(f"C-Adv Probs: {raw_probs}")
        print(f"C-Adv Predicted Class: {raw_class} ({label_map[raw_class]})")
    else:
        print("C-Adv not invoked because Model A rejected or matched identity is None.")

if __name__ == "__main__":
    main()
