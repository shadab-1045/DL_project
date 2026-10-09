import os
import cv2
import glob
import time
import json
import torch
import hashlib
import numpy as np
from torchvision import transforms

from src.runtime.live_pipeline_v4 import LiveInferencePipelineV4
from src.preprocessing.v4_live_parity import V4LiveParityAdapter

def hash_file(filepath):
    if not os.path.exists(filepath): return None
    with open(filepath, 'rb') as f: return hashlib.sha256(f.read()).hexdigest()

def run_regression():
    print("--- Phase 7F.21 Regression Validation ---")
    
    # Hashes Before
    h_v11 = hash_file('experiments/model_c_v4/best_model.pt')
    h_v13_before = hash_file('experiments/model_c_v4/best_fusion_model.pt')
    
    # Initialize Pipeline
    pipeline = LiveInferencePipelineV4(gallery_dir="experiments/live_demo/v4_gallery_physical")
    
    # Verify Threshold
    print(f"Model A Threshold: {pipeline.identity_pipeline.match_threshold}")
    assert abs(pipeline.identity_pipeline.match_threshold - 0.244529) < 1e-4
    
    # Enroll Identity
    ref_images = glob.glob("sample_data/lfw/Angelina_Jolie/*.jpg")[:5]
    pipeline.identity_pipeline.enroll("Alice", ref_images)
    src_img = cv2.imread(ref_images[0])
    faces = pipeline.identity_pipeline.engine.extract_faces(src_img)
    best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    pipeline.swapper.set_source_face(best_face)
    
    # Select Samples
    gen_frames = glob.glob("sample_data/lfw/Angelina_Jolie/*.jpg")[5:10]
    diff_frames = glob.glob("sample_data/lfw/Alejandro_Toledo/*.jpg")[:5]
    
    # Original (old) Transform for comparison
    old_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    results = {"genuine": [], "different": [], "impersonation": []}
    
    def evaluate(img_bgr, condition):
        pipeline.reset_temporal_state()
        
        # 1. Evaluate with Current (Parity) Pipeline
        res_new = pipeline.process_frame(img_bgr, condition=condition, frame_id=0)
        
        # 2. Evaluate with Old Pipeline Manually
        res_old = {"v4_p_synthetic": None, "final_ui_state": None, "raw_class": None}
        if res_new["model_a_accept"]:
            from src.preprocessing.alignment import align_face
            import torch.nn.functional as F
            
            # Reconstruct the exact inputs for old path
            proc_img = img_bgr
            if condition == "impersonation":
                # We need to swap manually to test old path
                tf = pipeline.identity_pipeline.engine.extract_faces(proc_img)
                proc_img = pipeline.swapper.process_frame(proc_img, tf)
                
            f2 = pipeline.identity_pipeline.engine.extract_faces(proc_img)
            bf2 = max(f2, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            aligned = align_face(proc_img, bf2.kps)
            
            t_old = old_transform(aligned).unsqueeze(0).to(pipeline.device)
            sim_old = torch.tensor([[res_new["model_a_similarity"]]], dtype=torch.float32).to(pipeline.device)
            
            with torch.no_grad():
                logits, ps, _ = pipeline.c_adv_v4(t_old, sim_old)
                res_old["v4_p_synthetic"] = ps.item()
                res_old["raw_class"] = int(torch.argmax(logits, dim=1).item())
                
                # UI State mapping
                if res_old["raw_class"] == 0: res_old["final_ui_state"] = "VERIFIED"
                elif res_old["raw_class"] == 2: res_old["final_ui_state"] = "SUSPECTED_IMPERSONATION"
                else: res_old["final_ui_state"] = "UNKNOWN"
                
        return {"new": res_new, "old": res_old}

    print("\n--- Evaluating Genuine ---")
    for f in gen_frames:
        img = cv2.imread(f)
        out = evaluate(img, "genuine")
        results["genuine"].append((f, out))
        print(f"Gen {os.path.basename(f)}: P(Synth) Old: {out['old']['v4_p_synthetic']} -> New: {out['new']['v4_p_synthetic']:.4f}")
        
    print("\n--- Evaluating Different Person ---")
    for f in diff_frames:
        img = cv2.imread(f)
        out = evaluate(img, "genuine")
        results["different"].append((f, out))
        print(f"Diff {os.path.basename(f)}: ModA Acc: {out['new']['model_a_accept']}")
        
    print("\n--- Evaluating Impersonation ---")
    for f in diff_frames:
        img = cv2.imread(f)
        out = evaluate(img, "impersonation")
        results["impersonation"].append((f, out))
        print(f"Imp {os.path.basename(f)}: P(Synth) Old: {out['old']['v4_p_synthetic']:.4f} -> New: {out['new']['v4_p_synthetic']:.4f}")

    h_v13_after = hash_file('experiments/model_c_v4/best_fusion_model.pt')
    assert h_v13_before == h_v13_after, "Checkpoint modified!"
    print("\n[PASS] Checkpoints intact.")
    
    with open('experiments/live_demo/v4/phase7f21_regression_data.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_regression()
