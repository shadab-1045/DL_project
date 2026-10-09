import os
import cv2
import glob
import numpy as np
from src.runtime.live_pipeline_v4 import LiveInferencePipelineV4

def test_v4_live_pipeline():
    print("--- V4 Live Pipeline Offline Smoke Test ---")
    
    # Init Pipeline
    pipeline = LiveInferencePipelineV4(gallery_dir="experiments/live_demo/v4_gallery_test")
    pipeline.start_session("test_session_001")
    
    # Enroll Angelina Jolie
    ref_images = glob.glob("sample_data/lfw/Angelina_Jolie/*.jpg")[:5]
    assert len(ref_images) > 0, "No reference images found!"
    
    print(f"Enrolling Alice (Angelina) with {len(ref_images)} images...")
    pipeline.identity_pipeline.enroll("Alice", ref_images)
    
    # Enroll Native InSwapper Source
    alice_img = cv2.imread(ref_images[0])
    faces = pipeline.identity_pipeline.engine.extract_faces(alice_img)
    best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    pipeline.swapper.set_source_face(best_face)
    
    # Test 1: Genuine Presentation
    print("\n--- Test 1: Genuine Presentation ---")
    gen_img = cv2.imread(ref_images[-1])
    res1 = pipeline.process_frame(gen_img, condition="genuine", frame_id=1)
    
    print(f"Model A Similarity: {res1['model_a_similarity']:.4f}")
    print(f"V4 P(Id_Match): {res1['v4_p_identity_match']:.4f}")
    print(f"V4 P(Synthetic): {res1['v4_p_synthetic']:.4f}")
    print(f"Raw Predicted Class: {res1['v4_predicted_class']}")
    print(f"Final UI State: {res1['final_ui_state']}")
    print(f"Total Latency: {res1['latency_total']:.4f}s (FPS: {res1['fps']:.1f})")
    
    # Test 2: Different Person Presentation
    print("\n--- Test 2: Different Person Presentation ---")
    diff_images = glob.glob("sample_data/lfw/Alejandro_Toledo/*.jpg")
    diff_img = cv2.imread(diff_images[0])
    res2 = pipeline.process_frame(diff_img, condition="genuine", frame_id=2)
    
    print(f"Model A Similarity: {res2['model_a_similarity']:.4f}")
    if 'v4_p_identity_match' in res2:
        print(f"V4 P(Id_Match): {res2['v4_p_identity_match']:.4f}")
    print(f"Final UI State: {res2['final_ui_state']}")
    
    # Test 3: Impersonation (Native InSwapper)
    print("\n--- Test 3: Impersonation Presentation ---")
    res3 = pipeline.process_frame(diff_img, condition="impersonation", frame_id=3)
    
    print(f"Swap Applied: {res3['swap_applied']}")
    print(f"Model A Similarity: {res3['model_a_similarity']:.4f}")
    print(f"V4 P(Id_Match): {res3.get('v4_p_identity_match', 0.0):.4f}")
    print(f"V4 P(Synthetic): {res3.get('v4_p_synthetic', 0.0):.4f}")
    print(f"Raw Predicted Class: {res3.get('v4_predicted_class', -1)}")
    print(f"Final UI State: {res3['final_ui_state']}")
    print(f"Latencies - Swap: {res3['latency_swap']:.4f}s, Detect: {res3['latency_detect_align']:.4f}s, Vis: {res3['latency_visual_branch']:.4f}s, Fus: {res3['latency_fusion']:.4f}s")
    
    pipeline.end_session()
    print("\n[PASS] Smoke validation complete.")

if __name__ == "__main__":
    test_v4_live_pipeline()
