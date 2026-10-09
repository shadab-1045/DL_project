import cv2
import numpy as np
import glob
from src.identity.arcface import IdentityEngine
from src.runtime.identity_pipeline import IdentityPipeline

engine = IdentityEngine()

# Load enrollment images
alice_images = glob.glob("sample_data/lfw/Angelina_Jolie/*.jpg")[:5]
enrollment_embs = []
for p in alice_images:
    img = cv2.imread(p)
    emb = engine.get_embedding(img)
    if emb is not None:
        enrollment_embs.append(emb)

# Load diagnostic image
diag_img = cv2.imread("experiments/live_demo/manual/diagnostic_genuine.jpg")
diag_emb = engine.get_embedding(diag_img)

if diag_emb is not None:
    print(f"Enrollment Embs: {len(enrollment_embs)}")
    for i, e in enumerate(enrollment_embs):
        sim = np.dot(e, diag_emb) / (np.linalg.norm(e) * np.linalg.norm(diag_emb))
        print(f"Sim {i+1}: {sim:.4f}")
    
    # Compute the mean embedding similarity
    mean_emb = np.mean(enrollment_embs, axis=0)
    mean_emb = mean_emb / np.linalg.norm(mean_emb)
    final_sim = np.dot(mean_emb, diag_emb) / (np.linalg.norm(mean_emb) * np.linalg.norm(diag_emb))
    print(f"Mean Emb Sim: {final_sim:.4f}")
else:
    print("No face detected in diagnostic image.")
