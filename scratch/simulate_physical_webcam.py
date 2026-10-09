import os
import cv2
import json
import glob
import time
import hashlib
from src.runtime.live_pipeline_v4 import LiveInferencePipelineV4

def hash_file(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def run_physical_emulation():
    print("--- V4 Physical Webcam Validation (Emulation) ---")
    
    # 1. Preflight Hashes
    h_v11 = hash_file('experiments/model_c_v4/best_model.pt')
    h_v12 = hash_file('experiments/model_c_v4/best_model_id_fixed.pt')
    h_v13 = hash_file('experiments/model_c_v4/best_fusion_model.pt')
    
    print(f"Preflight Check:")
    print(f"  V11 (Visual): {h_v11}")
    print(f"  V12 (Id Fixed): {h_v12}")
    print(f"  V13 (Fusion): {h_v13}")
    
    assert h_v11 == "0c5e8463023e2d603070ad4aec435be2bfc16c14115237942e38380983acd192"
    assert h_v12 == "ca9394dbb3e737e518de4aa77632c3e25b532abce5034e25cfe4a5d11c34837c"
    assert h_v13 == "f2647012bc2863d988e65f980310cac9b60a0beb8c3e068f38b564f4383be2f5"
    
    os.makedirs('experiments/live_demo/v4/physical', exist_ok=True)
    os.makedirs('experiments/live_demo/v4/physical/diagnostic_frames', exist_ok=True)
    
    pipeline = LiveInferencePipelineV4(gallery_dir="experiments/live_demo/v4_gallery_physical")
    print(f"  Model A Threshold: {pipeline.identity_pipeline.match_threshold}")
    assert abs(pipeline.identity_pipeline.match_threshold - 0.244529) < 1e-4
    
    # Enroll Identity (Angelina Jolie)
    ref_images = glob.glob("sample_data/lfw/Angelina_Jolie/*.jpg")[:5]
    pipeline.identity_pipeline.enroll("Alice", ref_images)
    
    # Set InSwapper Source
    src_img = cv2.imread(ref_images[0])
    faces = pipeline.identity_pipeline.engine.extract_faces(src_img)
    best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    pipeline.swapper.set_source_face(best_face)
    
    # Use remaining images as "live frames"
    gen_frames = glob.glob("sample_data/lfw/Angelina_Jolie/*.jpg")[5:15]
    diff_frames = glob.glob("sample_data/lfw/Alejandro_Toledo/*.jpg")[:10]
    
    # Test A: Genuine
    print("\n--- Test A: Genuine Physical Presentation ---")
    pipeline.start_session("physical_genuine")
    for i, path in enumerate(gen_frames):
        img = cv2.imread(path)
        res = pipeline.process_frame(img, condition="genuine", frame_id=i)
        print(f"Frame {i}: ModA Sim: {res['model_a_similarity']:.4f}, ModA Acc: {res['model_a_accept']}, P(Id): {res.get('v4_p_identity_match', 0):.4f}, P(Synth): {res.get('v4_p_synthetic', 0):.4f}, V4_Class: {res.get('v4_predicted_class', -1)}, UI: {res['final_ui_state']}")
        if i == 5:
            cv2.imwrite(f"experiments/live_demo/v4/physical/diagnostic_frames/genuine_frame_{i}.jpg", img)
    pipeline.end_session()
    
    # Test B: Different Person
    print("\n--- Test B: Different-Person Physical Presentation ---")
    pipeline.start_session("physical_different")
    for i, path in enumerate(diff_frames):
        img = cv2.imread(path)
        res = pipeline.process_frame(img, condition="genuine", frame_id=i)
        print(f"Frame {i}: ModA Sim: {res['model_a_similarity']:.4f}, ModA Acc: {res['model_a_accept']}, UI: {res['final_ui_state']}")
        if i == 5:
            cv2.imwrite(f"experiments/live_demo/v4/physical/diagnostic_frames/diff_frame_{i}.jpg", img)
    pipeline.end_session()
    
    # Test C: Impersonation
    print("\n--- Test C: Native InSwapper Impersonation ---")
    pipeline.start_session("physical_impersonation")
    for i, path in enumerate(diff_frames):
        img = cv2.imread(path)
        res = pipeline.process_frame(img, condition="impersonation", frame_id=i)
        print(f"Frame {i}: Swap OK: {res['swap_applied']}, ModA Sim: {res['model_a_similarity']:.4f}, ModA Acc: {res['model_a_accept']}, P(Id): {res.get('v4_p_identity_match', 0):.4f}, P(Synth): {res.get('v4_p_synthetic', 0):.4f}, UI: {res['final_ui_state']}")
        if i == 5:
            cv2.imwrite(f"experiments/live_demo/v4/physical/diagnostic_frames/imp_frame_{i}.jpg", pipeline.last_processed_frame)
    pipeline.end_session()
    
    # Merge logs to physical dir
    os.system("move experiments\\live_demo\\v4\\physical_*.jsonl experiments\\live_demo\\v4\\physical\\")
    print("Done.")

if __name__ == "__main__":
    run_physical_emulation()
