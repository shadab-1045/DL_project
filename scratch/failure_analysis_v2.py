import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from collections import defaultdict
from tqdm import tqdm
import json

from src.models.model_c import AntiImpersonationModel
from src.data.model_c_dataset import AntiImpersonationDataset
from src.identity.arcface import IdentityEngine
import cv2

def extract_features(model, loader, device):
    all_visual = []
    all_ref_embs = []
    all_sims = []
    all_labels = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for imgs, embs, labels in tqdm(loader, desc="Extracting"):
            imgs, embs = imgs.to(device), embs.to(device)
            visual_feat = model.backbone(imgs)
            visual_proj = model.visual_proj(visual_feat)
            sim = torch.cosine_similarity(visual_proj, embs, dim=1)
            
            fused = torch.cat([visual_feat, embs, sim.unsqueeze(1)], dim=1)
            logits = model.fusion(fused)
            probs = torch.nn.functional.softmax(logits, dim=1)
            
            all_visual.append(visual_feat.cpu().numpy())
            all_ref_embs.append(embs.cpu().numpy())
            all_sims.append(sim.cpu().numpy())
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())
            
    return (
        np.concatenate(all_visual),
        np.concatenate(all_ref_embs),
        np.concatenate(all_sims),
        np.concatenate(all_labels),
        np.concatenate(all_probs)
    )

def analyze():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AntiImpersonationModel().to(device)
    model.load_state_dict(torch.load('experiments/model_c_adv_v2/best_model.pt', map_location=device))
    
    train_ds = AntiImpersonationDataset('data/manifests/train_pairs_v2.csv')
    val_ds = AntiImpersonationDataset('data/manifests/val_pairs_v2.csv')
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    print("Extracting Train features...")
    tr_vis, tr_ref, tr_sim, tr_y, tr_prob = extract_features(model, train_loader, device)
    print("Extracting Val features...")
    va_vis, va_ref, va_sim, va_y, va_prob = extract_features(model, val_loader, device)
    
    # 1. CLASS FEATURE SEPARABILITY (Visual Backbone Features)
    print("\n--- 1. CLASS FEATURE SEPARABILITY ---")
    print(f"Feature Dimensionality: {tr_vis.shape[1]}")
    
    centroids = {}
    for c, name in enumerate(["Genuine", "DiffPerson", "Impersonation"]):
        mask = (va_y == c)
        if mask.sum() > 0:
            centroids[c] = np.mean(va_vis[mask], axis=0)
    
    print("Centroid Distances (L2):")
    if 0 in centroids and 1 in centroids:
        print(f"Genuine <-> DiffPerson: {np.linalg.norm(centroids[0] - centroids[1]):.4f}")
    if 0 in centroids and 2 in centroids:
        print(f"Genuine <-> Impersonation: {np.linalg.norm(centroids[0] - centroids[2]):.4f}")
    if 1 in centroids and 2 in centroids:
        print(f"DiffPerson <-> Impersonation: {np.linalg.norm(centroids[1] - centroids[2]):.4f}")
        
    print("Linear Probe on VISUAL features (Train -> Val)...")
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(tr_vis, tr_y)
    preds = clf.predict(va_vis)
    acc = accuracy_score(va_y, preds)
    print(f"Linear Probe Accuracy: {acc:.4f}")
    from sklearn.metrics import classification_report
    print(classification_report(va_y, preds, target_names=["Gen", "Diff", "Imp"], zero_division=0))

    # 2. IDENTITY-SIMILARITY ANALYSIS (Internal Model Sim)
    print("\n--- 2. IDENTITY-SIMILARITY ANALYSIS (Model Internal) ---")
    for c, name in enumerate(["Genuine", "DiffPerson", "Impersonation"]):
        mask = (va_y == c)
        if mask.sum() > 0:
            sims = va_sim[mask]
            print(f"{name} Sim - Mean: {np.mean(sims):.4f}, Std: {np.std(sims):.4f}, Min: {np.min(sims):.4f}, Max: {np.max(sims):.4f}")

    # Now let's calculate REAL ArcFace similarity using IdentityEngine
    print("\n--- REAL ARCFACE SIMILARITY (Model A) ---")
    engine = IdentityEngine()
    
    def get_real_sims(ds):
        sims = {0: [], 1: [], 2: []}
        for pair in ds.pairs:
            probe_path = pair['probe_path']
            ref_path = pair['reference_embedding_path']
            lbl = ds.label_map[pair['label']]
            
            img = cv2.imread(probe_path)
            
            # Since the image is already aligned 112x112, we can extract embedding directly!
            # Bypass detection pipeline because it fails on tight crops.
            rec_model = None
            for m in engine.app.models.values():
                if m.taskname == 'recognition':
                    rec_model = m
                    break
                    
            probe_emb = rec_model.get_feat(img).flatten()
            
            # Normalize embedding
            probe_emb = probe_emb / np.linalg.norm(probe_emb)
            ref_emb = np.load(ref_path)
            # cosine sim
            sim = np.dot(probe_emb, ref_emb) / (np.linalg.norm(probe_emb) * np.linalg.norm(ref_emb))
            sims[lbl].append(sim)
        return sims
        
    real_sims = get_real_sims(val_ds)
    for c, name in enumerate(["Genuine", "DiffPerson", "Impersonation"]):
        if real_sims[c]:
            arr = np.array(real_sims[c])
            print(f"{name} Real ArcFace Sim - Mean: {np.mean(arr):.4f}, Std: {np.std(arr):.4f}")

    # 4. TRAIN VS VAL DISTRIBUTION CHECK
    print("\n--- 4. TRAIN VS VAL DISTRIBUTION CHECK ---")
    print(f"Train Internal Sim Mean: {np.mean(tr_sim):.4f} | Val Internal Sim Mean: {np.mean(va_sim):.4f}")
    
    for c, name in enumerate(["Genuine", "DiffPerson", "Impersonation"]):
        tr_mask = (tr_y == c)
        va_mask = (va_y == c)
        print(f"[{name}] Train Sim: {np.mean(tr_sim[tr_mask]):.4f} | Val Sim: {np.mean(va_sim[va_mask]):.4f}")

if __name__ == "__main__":
    analyze()
