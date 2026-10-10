"""Enroll the person who will stand in front of the camera (the "genuine user"), WITHOUT changing the face-swap source.
Run this before the demo, in the same lighting/camera you will demo with, then restart the server.
  python enroll_user.py Arsal --webcam          # 6 snapshots, 1 s apart (look at the camera, vary your pose slightly)
  python enroll_user.py Arsal --webcam 1        # same, using camera index 1 (try 0, 1, 2 if the default is black)
  python enroll_user.py Arsal photo1.jpg photo2.jpg ..."""
import os, sys, time, tempfile
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__)); DL = os.path.dirname(ROOT)   # repository root (the DL_project code base)
if len(sys.argv) < 3: sys.exit(__doc__)
name, args = sys.argv[1], sys.argv[2:]
paths = []
if args[0] == "--webcam":
    cam = int(args[1]) if len(args) > 1 else 0
    cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW if os.name == "nt" else 0)
    if not cap.isOpened(): sys.exit(f"Cannot open camera {cam}.")
    tmp = tempfile.mkdtemp(); time.sleep(1.5)  # let auto-exposure settle
    for i in range(6):
        ok, frame = cap.read()
        if ok and frame.mean() < 10: cap.release(); sys.exit(f"Camera {cam} returns black frames (mean {frame.mean():.1f}/255): open the privacy shutter / camera key, close other apps using the camera, or try another index.")
        if ok: p = os.path.join(tmp, f"{i}.jpg"); cv2.imwrite(p, frame); paths.append(p); print(f"captured {i + 1}/6"); time.sleep(1)
    cap.release()
else:
    paths = [os.path.abspath(a) for a in args]   # absolute before we chdir to the repository root

os.chdir(DL); sys.path.insert(0, DL)
from src.runtime.identity_pipeline import IdentityPipeline
pipe = IdentityPipeline(gallery_dir="experiments/live_demo/v4_gallery")
if not pipe.enroll(name, paths): sys.exit("Enrollment failed: no face detected in the images.")
print(f"Enrolled '{name}' from {len(paths)} image(s) into experiments/live_demo/v4_gallery (repository root). Restart the server.")
