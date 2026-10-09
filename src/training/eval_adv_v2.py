import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import json
from tqdm import tqdm

from src.models.model_c import AntiImpersonationModel
from src.data.model_c_dataset import AntiImpersonationDataset
from src.training.adversarial import fgsm_attack

def evaluate_robustness():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AntiImpersonationModel().to(device)
    model.load_state_dict(torch.load('experiments/model_c_adv_v2/best_model.pt', map_location=device))
    
    val_ds = AntiImpersonationDataset('data/manifests/val_pairs_v2.csv')
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    
    epsilons = [0.0, 0.01, 0.05, 0.10]
    
    results = {}
    
    # Check baseline (no attack) first
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for imgs, embs, labels in tqdm(val_loader, desc=f"Eval eps=0.0"):
            imgs, embs, labels = imgs.to(device), embs.to(device), labels.to(device)
            logits = model(imgs, embs)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1, 2], zero_division=0)
    macro_f1 = np.mean(f1)
    
    results[0.0] = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": precision_recall_fscore_support(all_labels, all_preds, average='weighted', labels=[0, 1, 2], zero_division=0)[2],
        "genuine_recall": r[0],
        "different_recall": r[1],
        "impersonation_recall": r[2],
        "impersonation_precision": p[2],
        "impersonation_f1": f1[2],
        "per_class_precision": p.tolist(),
        "per_class_recall": r.tolist(),
        "per_class_f1": f1.tolist()
    }
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    print(f"\n--- EPSILON 0.0 ---")
    print(f"Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}")
    print(f"Imp Recall: {r[2]:.4f} | Imp Precision: {p[2]:.4f}")
    print("Confusion Matrix (G, D, I):")
    print(cm)

    # Now attacks
    for eps in epsilons[1:]:
        model.eval()
        all_preds = []
        all_labels = []
        for imgs, embs, labels in tqdm(val_loader, desc=f"Eval eps={eps}"):
            imgs, embs, labels = imgs.to(device), embs.to(device), labels.to(device)
            
            perturbed_imgs = fgsm_attack(model, imgs, embs, labels, criterion, eps)
            
            with torch.no_grad():
                logits = model(perturbed_imgs, embs)
                preds = torch.argmax(logits, dim=1)
                
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        acc = accuracy_score(all_labels, all_preds)
        p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1, 2], zero_division=0)
        macro_f1 = np.mean(f1)
        
        results[eps] = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "impersonation_precision": p[2],
            "impersonation_recall": r[2],
            "impersonation_f1": f1[2]
        }
        
        print(f"\n--- EPSILON {eps} ---")
        print(f"Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}")
        print(f"Imp Recall: {r[2]:.4f} | Imp Precision: {p[2]:.4f} | Imp F1: {f1[2]:.4f}")

    with open('experiments/model_c_adv_v2/val_robustness.json', 'w') as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    evaluate_robustness()
