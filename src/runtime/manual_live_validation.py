import cv2
import time
import os
import sys
import glob
import json
import hashlib
from src.runtime.live_pipeline import LiveInferencePipeline

EXPECTED_HASHES = {
    "Model A": {
        "path": r"C:\Users\shada\.insightface\models\buffalo_l\w600k_r50.onnx",
        "hash": "4C06341C33C2CA1F86781DAB0E829F88AD5B64BE9FBA56E56BC9EBDEFC619E43"
    },
    "C-Control": {
        "path": "experiments/model_c_control/best_model.pt",
        "hash": "1CF70BA964B5E60C6632AC8532A7BA8BCEC9F4509E425CE9CE6DF71E00C86E51"
    },
    "C-Adv": {
        "path": "experiments/model_c_adv/best_model.pt",
        "hash": "A562487A5B3CC085E3CC1EED6FBEA93748376FB75BBD2FBDF3D0E871F966AFBD"
    }
}

def compute_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest().upper()

def run_pre_flight_checks(pipeline, alice_images):
    print("--- PRE-FLIGHT CHECKS ---")
    
    # 1. Hashes
    for model_name, info in EXPECTED_HASHES.items():
        if not os.path.exists(info["path"]):
            print(f"[FAIL] Missing model file: {info['path']}")
            return False
        h = compute_sha256(info["path"])
        if h != info["hash"]:
            print(f"[FAIL] Hash mismatch for {model_name}. Expected: {info['hash']}, Got: {h}")
            return False
    print("[PASS] Phase 5 artifact hashes match.")
    
    # 2. Threshold check
    if abs(pipeline.identity_pipeline.match_threshold - 0.244529) > 1e-6:
        print(f"[FAIL] Model A Threshold is not exactly 0.244529! Found {pipeline.identity_pipeline.match_threshold}")
        return False
    print("[PASS] Model A threshold is exactly 0.244529.")
    
    # 3. Source face exists and detectable
    if not alice_images or not os.path.exists(alice_images[0]):
        print(f"[FAIL] Source face image not found.")
        return False
    alice_img = cv2.imread(alice_images[0])
    alice_faces = pipeline.identity_pipeline.engine.extract_faces(alice_img)
    if not alice_faces:
        print(f"[FAIL] Failed to detect face in source image.")
        return False
    print("[PASS] Source face detected successfully.")
    
    # 4. InSwapper, ArcFace, C-Adv are loaded implicitly by LiveInferencePipeline init.
    if pipeline.swapper is None or pipeline.c_adv is None:
        print(f"[FAIL] Pipeline failed to initialize components.")
        return False
    print("[PASS] Components initialized.")
    
    # 5. Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FAIL] Cannot open physical webcam.")
        return False
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[FAIL] Cannot capture frames from webcam.")
        cap.release()
        return False
    cap.release()
    print("[PASS] Webcam captures successfully.")
    
    print("Pre-flight checks passed.\n")
    return alice_faces

def draw_overlay(frame, res, condition_text):
    h, w = frame.shape[:2]
    # Background panel
    cv2.rectangle(frame, (0, 0), (w, 180), (0, 0, 0), -1)
    
    color = (255, 255, 255)
    cv2.putText(frame, f"Cond: {condition_text}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"FPS: {res.get('fps', 0):.1f}", (w - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.putText(frame, f"Raw Sim: {res.get('raw_model_a_similarity', 0):.3f} (Thresh: {res.get('model_a_threshold', 0.244):.3f})", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    cv2.putText(frame, f"Raw Accepted: {res.get('model_a_raw_accept', False)}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    
    probs = res.get("raw_c_probs", [0.0, 0.0, 0.0])
    class_str = ["Genuine", "Diff", "Impersonation"][res.get("raw_c_class", 1)] if probs else "N/A"
    cv2.putText(frame, f"C-Adv Class: {class_str} | Probs: {[f'{p:.2f}' for p in probs]}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    
    state_color = (0, 255, 0) if res.get("final_state") == "VERIFIED" else (0, 0, 255) if res.get("final_state") == "SUSPECTED_IMPERSONATION" else (0, 165, 255)
    cv2.putText(frame, f"Final State: {res.get('final_state', 'UNKNOWN')}", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2)
    
    cv2.putText(frame, "Controls: [G] Genuine | [I] Impersonation | [Q] Quit", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

def run_manual_validation():
    print("Initializing Live Pipeline (Native InSwapper)...")
    pipeline = LiveInferencePipeline()
    # Force manual log directory
    os.makedirs("experiments/live_demo/manual", exist_ok=True)
    
    alice_name = "Alice"
    alice_images = glob.glob("sample_data/lfw/Angelina_Jolie/*.jpg")[:5]
    
    alice_faces = run_pre_flight_checks(pipeline, alice_images)
    if not alice_faces:
        sys.exit(1)
        
    print(f"Enrolling {alice_name}...")
    pipeline.identity_pipeline.enroll(alice_name, alice_images)
    
    alice_face = max(alice_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    pipeline.swapper.set_source_face(alice_face)
    print("Source face (Alice) set in Native InSwapper.")
    
    print("\n--- MANUAL LIVE VALIDATION ---")
    
    stats = {"frames": 0, "faces": 0, "accepts": 0, "c_adv": 0, "VERIFIED": 0, "UNKNOWN": 0, "SUSPECTED_IMPERSONATION": 0, "total_latency": 0.0}
    frame_id = 0
    
    while True:
        print("\nControls:")
        print("[G] Run Genuine Condition (10 frames)")
        print("[D] Run Different-Person Condition (10 frames)")
        print("[I] Run Impersonation Condition (10 frames)")
        print("[Q] Quit")
        choice = input("Select an option: ").strip().lower()
        
        if choice == 'q':
            break
        elif choice == 'g':
            current_condition = "genuine"
        elif choice == 'd':
            current_condition = "different_person"
        elif choice == 'i':
            print("\n*** IMPORTANT DIRECTION VERIFICATION ***")
            print("SOURCE: Enrolled identity (Alice)")
            print("TARGET: Live physical participant")
            print("Diagnostic frames will be saved to verify Alice's face on the participant's body.")
            current_condition = "impersonation"
        else:
            print("Invalid choice.")
            continue
            
        print(f"\nStarting webcam for 10 frames of {current_condition.upper()}...")
        cap = cv2.VideoCapture(0)
        
        if pipeline.log_file: pipeline.log_file.close()
        pipeline.start_session(f"manual_{current_condition}")
        pipeline.log_file = open(os.path.join("experiments/live_demo/manual", f"manual_{current_condition}_{int(time.time())}.jsonl"), 'w')
        
        for i in range(10):
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame.")
                break
                
            start = time.time()
            res = pipeline.process_frame(frame, condition=current_condition, frame_id=frame_id)
            
            # Save diagnostic frame on the 5th iteration
            if i == 5:
                display_frame = frame.copy()
                if current_condition == "impersonation":
                    faces = pipeline.identity_pipeline.engine.extract_faces(frame)
                    if faces:
                        display_frame = pipeline.swapper.process_frame(frame, faces)
                
                # Draw minimal info
                cv2.putText(display_frame, f"Cond: {current_condition.upper()}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Sim: {res.get('raw_model_a_similarity', 0):.3f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"State: {res.get('final_state', 'UNKNOWN')}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                out_path = os.path.join("experiments/live_demo/manual", f"diagnostic_{current_condition}.jpg")
                cv2.imwrite(out_path, display_frame)
                print(f"--> Saved diagnostic frame to {out_path}. Please inspect it.")
            
            # Print frame summary to terminal
            print(f"Frame {frame_id:04d} | Face: {res.get('face_detected')} | Sim: {res.get('raw_model_a_similarity', 0):.3f} | State: {res.get('final_state', 'UNKNOWN')}")
            
            # Update Stats
            stats["frames"] += 1
            if res.get("face_detected"): stats["faces"] += 1
            if res.get("model_a_raw_accept"): stats["accepts"] += 1
            if res.get("raw_c_probs", [0,0,0])[0] != 0.0 or res.get("raw_c_probs", [0,0,0])[1] != 0.0: stats["c_adv"] += 1
            stats[res.get("final_state", "UNKNOWN")] += 1
            stats["total_latency"] += res.get("total_latency", 0)
            
            frame_id += 1
            
        cap.release()
        if pipeline.log_file:
            pipeline.log_file.close()

    print("\n--- SESSION SUMMARY ---")
    print(f"Total Frames: {stats['frames']}")
    print(f"Faces Detected: {stats['faces']}")
    print(f"Model A Accepts: {stats['accepts']}")
    print(f"C-Adv Invocations: {stats['c_adv']}")
    print(f"VERIFIED: {stats['VERIFIED']}")
    print(f"UNKNOWN: {stats['UNKNOWN']}")
    print(f"SUSPECTED_IMPERSONATION: {stats['SUSPECTED_IMPERSONATION']}")
    avg_latency = (stats['total_latency'] / max(1, stats['frames']))
    print(f"Average Latency: {avg_latency:.3f}s")
    print(f"Approximate FPS: {1.0 / max(0.001, avg_latency):.1f}")
    print("\nManual live validation complete.")

if __name__ == "__main__":
    run_manual_validation()
