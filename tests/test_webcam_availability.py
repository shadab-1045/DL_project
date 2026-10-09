import cv2

def check_webcam():
    print("Attempting to open webcam 0...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("BLOCKER: Webcam 0 could not be opened. No hardware access available in this environment.")
        return False
    
    ret, frame = cap.read()
    if not ret:
        print("BLOCKER: Webcam opened but failed to read a frame.")
        cap.release()
        return False
        
    print(f"SUCCESS: Read frame of shape {frame.shape}")
    cap.release()
    return True

if __name__ == "__main__":
    check_webcam()
