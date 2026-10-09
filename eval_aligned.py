import os
import csv
import torch
import cv2
from src.models.model_c import AntiImpersonationModel
from src.preprocessing.alignment import align_face
from torchvision import transforms
import insightface
from tqdm import tqdm
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def check_full_confusion_matrix():
    c_adv = AntiImpersonationModel()
    c_adv.load_state_dict(torch.load("experiments/model_c_adv/best_model.pt", map_location=device))
    c_adv.to(device)
    c_adv.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    with open("data/manifests/test_pairs.csv", "r") as f:
        reader = csv.DictReader(f)
        test_pairs = list(reader)
        
    # matrix[actual][predicted]
    matrix = {
        "genuine": {"genuine": 0, "different_person": 0, "impersonation": 0},
        "different_person": {"genuine": 0, "different_person": 0, "impersonation": 0},
        "impersonation": {"genuine": 0, "different_person": 0, "impersonation": 0}
    }
    
    pred_map = {0: "genuine", 1: "different_person", 2: "impersonation"}
    
    for pair in tqdm(test_pairs):
        actual_label = pair["label"]
        probe_path = pair["probe_path"]
        emb_path = pair["reference_embedding_path"]
        
        img = cv2.imread(probe_path)
        if img is None:
            continue
            
        if actual_label == "impersonation":
            # These are context crops and must be aligned to simulate the live pipeline!
            faces = app.get(img)
            if not faces:
                continue
            best_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            aligned_rgb = align_face(img, best_face.kps)
        else:
            # Genuine and different_person are ALREADY aligned in the training set (112x112)!
            # Just convert to RGB as align_face would.
            aligned_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        face_tensor = transform(aligned_rgb).unsqueeze(0).to(device)
        
        ref_emb = np.load(emb_path)
        ref_tensor = torch.tensor(ref_emb, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = c_adv(face_tensor, ref_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))
            
        pred_label = pred_map[pred_class]
        matrix[actual_label][pred_label] += 1
            
    print("Confusion Matrix:")
    print(f"             Predicted")
    print(f"             G    D    I")
    print(f"Actual G     {matrix['genuine']['genuine']:<4} {matrix['genuine']['different_person']:<4} {matrix['genuine']['impersonation']:<4}")
    print(f"Actual D     {matrix['different_person']['genuine']:<4} {matrix['different_person']['different_person']:<4} {matrix['different_person']['impersonation']:<4}")
    print(f"Actual I     {matrix['impersonation']['genuine']:<4} {matrix['impersonation']['different_person']:<4} {matrix['impersonation']['impersonation']:<4}")

if __name__ == "__main__":
    check_full_confusion_matrix()
