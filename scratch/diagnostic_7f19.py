import os
import cv2
import json
import glob
import torch
import hashlib
import numpy as np
from PIL import Image
from torchvision import transforms
from src.preprocessing.alignment import align_face
from src.runtime.live_pipeline_v4 import LiveInferencePipelineV4
from src.models.model_c_v4 import ModelCV4

def hash_file(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def hash_img(img_array):
    if img_array is None:
        return None
    return hashlib.sha256(np.ascontiguousarray(img_array)).hexdigest()

def run_diagnostic():
    print("--- Phase 7F.19 Diagnostic ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelCV4().to(device)
    model.load_state_dict(torch.load('experiments/model_c_v4/best_fusion_model.pt', map_location=device))
    model.eval()

    # Log file parsing
    genuine_logs = []
    with open('experiments/live_demo/v4/physical/physical_genuine.jsonl', 'r') as f:
        for line in f:
            genuine_logs.append(json.loads(line))
            
    imp_logs = []
    with open('experiments/live_demo/v4/physical/physical_impersonation.jsonl', 'r') as f:
        for line in f:
            imp_logs.append(json.loads(line))
            
    # Reconstruct Genuine
    gen_frames = glob.glob("sample_data/lfw/Angelina_Jolie/*.jpg")[5:15]
    print("\n1. Reconstruct Genuine False-Positive Path")
    for i, path in enumerate(gen_frames):
        if i >= len(genuine_logs): break
        log = genuine_logs[i]
        h = hash_file(path)
        print(f"Sample {i}: {path} (Hash: {h[:8]})")
        print(f"  Swap Applied: {log.get('swap_applied', False)}")
        print(f"  ModA Cosine: {log['model_a_similarity']:.4f} (Accepted: {log['model_a_accept']})")
        print(f"  V4 P(Id): {log['v4_p_identity_match']:.4f}, V4 P(Synth): {log['v4_p_synthetic']:.4f}")
        print(f"  V4 Fusion Probs (EMA): {log['ema_c_probs']}")
        print(f"  Raw Class: {log['v4_predicted_class']}, UI State: {log['final_ui_state']}")
        
    print("\n3. Controlled Input Comparison (Genuine Frame 5)")
    img_path = gen_frames[0] # Frame 0 (Index 0 in log)
    img_bgr = cv2.imread(img_path)
    
    pipeline = LiveInferencePipelineV4(gallery_dir="experiments/live_demo/v4_gallery_physical")
    faces = pipeline.identity_pipeline.engine.extract_faces(img_bgr)
    best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    
    # A. Original Image
    print(f"  Original Image Hash: {hash_img(img_bgr)[:8]}")
    
    # C. Face Aligned
    aligned_rgb = align_face(img_bgr, best_face.kps)
    print(f"  Aligned RGB Hash: {hash_img(aligned_rgb)[:8]}")
    
    # D. Transformed Tensor (Live pipeline equivalent)
    t_live = pipeline.transform(aligned_rgb).unsqueeze(0).to(device)
    print(f"  Live Tensor Hash: {hash_img(t_live.cpu().numpy())[:8]}")
    
    # E. Emulate the Dataset pipeline
    # The dataset reads a 112x112 JPEG from disk. Let's simulate that by saving aligned_rgb to JPEG then reading with PIL.
    cv2.imwrite('scratch/temp_aligned.jpg', cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR))
    pil_dataset_sim = Image.open('scratch/temp_aligned.jpg').convert('RGB')
    transform_ds = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    t_ds = transform_ds(pil_dataset_sim).unsqueeze(0).to(device)
    print(f"  Dataset (JPEG) Tensor Hash: {hash_img(t_ds.cpu().numpy())[:8]}")
    
    # Compare
    with torch.no_grad():
        p_synth_live = model.visual_branch(t_live).item()
        p_synth_ds = model.visual_branch(t_ds).item()
        
    print(f"  Live P(Synth): {p_synth_live:.4f}")
    print(f"  Dataset-Emulated P(Synth): {p_synth_ds:.4f}")
    
    print("\n4. Inspect Impersonation Case")
    diff_frames = glob.glob("sample_data/lfw/Alejandro_Toledo/*.jpg")[:10]
    imp_log = imp_logs[0]
    imp_path = diff_frames[0]
    print(f"Sample 0 Impersonation: {imp_path}")
    print(f"  Swap Applied: {imp_log.get('swap_applied', False)}")
    print(f"  ModA Cosine: {imp_log['model_a_similarity']:.4f} (Accepted: {imp_log['model_a_accept']})")
    print(f"  V4 P(Id): {imp_log['v4_p_identity_match']:.4f}, V4 P(Synth): {imp_log['v4_p_synthetic']:.4f}")
    print(f"  V4 Fusion Probs (EMA): {imp_log['ema_c_probs']}")
    print(f"  Raw Class: {imp_log['v4_predicted_class']}, UI State: {imp_log['final_ui_state']}")

if __name__ == "__main__":
    run_diagnostic()
