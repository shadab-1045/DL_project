import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import json
from tqdm import tqdm

from src.models.model_c_v3 import AntiImpersonationModelV3
from src.data.model_c_v3_dataset import AntiImpersonationV3Dataset
from src.training.train_model_c_adv_v3 import fgsm_attack_v3

def evaluate_robustness():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AntiImpersonationModelV3().to(device)
    model.load_state_dict(torch.load('experiments/model_c_adv_v3/best_model.pt', map_location=device))
    
    val_ds = AntiImpersonationV3Dataset('data/manifests/val_pairs_v2.csv')
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    
    epsilons = [0.0, 0.01, 0.05, 0.10]
    
    results = {}
    
    for eps in epsilons:
        model.eval()
        all_preds, all_labels = [], []
        
        for imgs, ref_embs, probe_embs, sims, labels in tqdm(val_loader, desc=f"Eval eps={eps}"):
            imgs, ref_embs, probe_embs, sims, labels = imgs.to(device), ref_embs.to(device), probe_embs.to(device), sims.to(device), labels.to(device)
            
            if eps > 0.0:
                perturbed_imgs = fgsm_attack_v3(model, imgs, ref_embs, probe_embs, sims, labels, criterion, eps)
            else:
                perturbed_imgs = imgs
                
            with torch.no_grad():
                logits = model(perturbed_imgs, ref_embs, probe_embs, sims)
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

    with open('experiments/model_c_adv_v3/val_robustness.json', 'w') as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    evaluate_robustness()
