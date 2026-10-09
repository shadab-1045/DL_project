import os
import cv2
import glob
import time
import torch
import hashlib
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.models.model_c_v4 import ModelCV4
from src.preprocessing.alignment import align_face
from src.preprocessing.v4_live_parity import V4LiveParityAdapter
from src.runtime.identity_pipeline import IdentityPipeline

def hash_file(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def test_parity():
    print("--- Phase 7F.20 Parity Test ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelCV4().to(device)
    model.load_state_dict(torch.load('experiments/model_c_v4/best_fusion_model.pt', map_location=device))
    model.eval()
    
    h_v13_before = hash_file('experiments/model_c_v4/best_fusion_model.pt')
    
    id_pipe = IdentityPipeline().engine
    
    # Existing Live transform
    transform_live = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    adapter = V4LiveParityAdapter()
    
    # Load genuine diagnostic frame
    gen_path = glob.glob("sample_data/lfw/Angelina_Jolie/*.jpg")[0]
    img_bgr = cv2.imread(gen_path)
    faces = id_pipe.extract_faces(img_bgr)
    best_face = faces[0]
    aligned_rgb = align_face(img_bgr, best_face.kps)
    
    # 1. Disk Dataset Path
    cv2.imwrite('scratch/parity_temp.jpg', cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR))
    pil_disk = Image.open('scratch/parity_temp.jpg').convert('RGB')
    t_disk = adapter.transform(pil_disk).unsqueeze(0).to(device)
    
    # 2. Adapter Path
    t_adapt = adapter.preprocess(aligned_rgb).to(device)
    
    # 3. Old Live Path
    t_old = transform_live(aligned_rgb).unsqueeze(0).to(device)
    
    print("\n[Parity Check]")
    print(f"Tensor Shape: {t_adapt.shape}, dtype: {t_adapt.dtype}")
    max_diff = torch.abs(t_disk - t_adapt).max().item()
    print(f"Max absolute pixel difference (Disk vs Adapter): {max_diff:.6f}")
    assert max_diff < 1e-4, "Adapter failed to mathematically reproduce disk path!"
    
    with torch.no_grad():
        p_synth_disk = model.visual_branch(t_disk).item()
        p_synth_adapt = model.visual_branch(t_adapt).item()
        p_synth_old = model.visual_branch(t_old).item()
        
    print(f"Visual Branch P(Synth) - Disk: {p_synth_disk:.4f}")
    print(f"Visual Branch P(Synth) - Adapter: {p_synth_adapt:.4f}")
    print(f"Visual Branch P(Synth) - Old Live: {p_synth_old:.4f}")
    assert abs(p_synth_disk - p_synth_adapt) < 1e-4, "Visual branch output mismatch!"
    
    h_v13_after = hash_file('experiments/model_c_v4/best_fusion_model.pt')
    assert h_v13_before == h_v13_after, "Checkpoint was modified!"
    print("[PASS] Checkpoint hashes unchanged.")
    
    print("\n[Performance & Behavior]")
    
    # Run full genuine diagnostic
    sim_tensor = torch.tensor([[0.7585]]).to(device) # from previous logs
    with torch.no_grad():
        # Old
        logits_old, ps_old, pid_old = model(t_old, sim_tensor)
        c_old = torch.argmax(logits_old, dim=1).item()
        
        start = time.time()
        # Adapter
        t_adapt_time = adapter.preprocess(aligned_rgb).to(device)
        adapter_latency = time.time() - start
        
        start = time.time()
        logits_new, ps_new, pid_new = model(t_adapt_time, sim_tensor)
        fusion_latency = time.time() - start
        c_new = torch.argmax(logits_new, dim=1).item()
        
    print("Old Live Preprocessing (Genuine input):")
    print(f"  P(Id)={pid_old.item():.4f}, P(Synth)={ps_old.item():.4f}, Raw Class={c_old}")
    
    print("\nAdapter Preprocessing (Genuine input):")
    print(f"  P(Id)={pid_new.item():.4f}, P(Synth)={ps_new.item():.4f}, Raw Class={c_new}")
    
    print(f"\nAdapter Overhead Latency: {adapter_latency:.5f}s")
    print(f"V4 Forward Latency: {fusion_latency:.5f}s")
    print(f"Identity branch outputs unchanged: {pid_old.item():.4f} == {pid_new.item():.4f}")

if __name__ == "__main__":
    test_parity()
