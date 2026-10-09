import cv2
import time
import os
import glob
from src.runtime.live_pipeline import LiveInferencePipeline

def run_live_validation():
    print("Initializing Live Pipeline...")
    pipeline = LiveInferencePipeline()
    
    # ENROLL ALICE
    alice_name = "Alice"
    alice_images = glob.glob("sample_data/lfw/Angelina_Jolie/*.jpg")[:5]
    if not alice_images:
        print("Could not find Angelina Jolie images to enroll Alice.")
        return
        
    print(f"Enrolling {alice_name} with {len(alice_images)} images...")
    pipeline.identity_pipeline.enroll(alice_name, alice_images)
    
    # Get Alice's source face for the Native InSwapper
    alice_img = cv2.imread(alice_images[0])
    alice_faces = pipeline.identity_pipeline.engine.extract_faces(alice_img)
    if not alice_faces:
        print("Failed to detect face in Alice's source image.")
        return
    alice_face = max(alice_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    pipeline.swapper.set_source_face(alice_face)
    print("Source face (Alice) set in Native InSwapper.")
    
    print("\n--- STARTING LIVE VALIDATION ---")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open physical webcam.")
        return
        
    # We will test two conditions: genuine and impersonation
    # But since the physical webcam is "Bob", Condition 1 (Genuine Alice) cannot be easily tested 
    # unless we hold up a picture of Alice, or we test "Unknown Bob" instead. 
    # Actually, if Bob is at the webcam, and no swap is applied, the expected state is UNKNOWN 
    # because Bob != Alice.
    # We will rename condition "genuine" to "baseline_unknown" for Bob.
    # And "impersonation" will be Alice swapped onto Bob.
    
    conditions = [
        {"name": "baseline_unknown", "frames": 10, "swap": "genuine"}, # 'genuine' means no swap
        {"name": "impersonation", "frames": 10, "swap": "impersonation"} # apply swap
    ]
    
    for cond in conditions:
        print(f"\nRunning Condition: {cond['name']}")
        pipeline.start_session(f"val_{cond['name']}")
        
        for i in range(cond['frames']):
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to read frame.")
                continue
                
            res = pipeline.process_frame(frame, condition=cond['swap'], frame_id=i)
            print(f"Frame {i:02d} | C:{cond['name']} | Sim: {res.get('ema_similarity', 0):.3f} | "
                  f"State: {res.get('final_state', 'ERR')} | FPS: {res.get('fps', 0):.1f}")
                  
        pipeline.end_session()
        time.sleep(1)
        
    cap.release()
    print("\nLive validation complete.")

if __name__ == "__main__":
    run_live_validation()
