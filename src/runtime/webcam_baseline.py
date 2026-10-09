import cv2
import time
from src.runtime.identity_pipeline import IdentityPipeline

def run_webcam():
    print("Loading pipeline...")
    pipeline = IdentityPipeline()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam demo. Press 'q' to quit.")
    
    frame_count = 0
    start_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run inference
        res = pipeline.identify(frame)
        
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()

        # Display results
        text_id = f"Identity: {res['identity'] if res['identity'] else 'UNKNOWN'}"
        text_sim = f"Similarity: {res['similarity']:.2f}"
        text_status = f"Status: {res['status']}"
        text_fps = f"FPS: {fps:.1f}"

        color = (0, 255, 0) if res['status'] == 'VERIFIED' else (0, 0, 255)
        
        cv2.putText(frame, text_id, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, text_sim, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, text_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, text_fps, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Webcam Identity Baseline", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()
