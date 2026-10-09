import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import json
from tqdm import tqdm

from src.models.model_visual_only import VisualOnlyModel
from src.training.train_model_visual_only import VisualOnlyDataset

def evaluate_visual_only():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VisualOnlyModel().to(device)
    model.load_state_dict(torch.load('experiments/model_visual_only/best_model.pt', map_location=device))
    model.eval()
    
    val_ds = VisualOnlyDataset('data/manifests/val_pairs_v2.csv')
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
    
    all_preds, all_labels = [], []
    
    for imgs, labels in tqdm(val_loader, desc=f"Evaluating"):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1, 2], zero_division=0)
    macro_f1 = np.mean(f1)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    
    unique, counts = np.unique(all_preds, return_counts=True)
    pred_counts = dict(zip([int(x) for x in unique], [int(x) for x in counts]))
    
    results = {
        "accuracy": float(acc),
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(np.average(f1, weights=np.bincount(all_labels))),
        "per_class": {
            "genuine": {"precision": float(p[0]), "recall": float(r[0]), "f1": float(f1[0])},
            "different_person": {"precision": float(p[1]), "recall": float(r[1]), "f1": float(f1[1])},
            "impersonation": {"precision": float(p[2]), "recall": float(r[2]), "f1": float(f1[2])}
        },
        "confusion_matrix": cm.tolist(),
        "pred_counts": pred_counts
    }
    
    with open('experiments/model_visual_only/val_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}")
    print(f"Genuine Recall: {r[0]:.4f}")
    print(f"Different Person Recall: {r[1]:.4f}")
    print(f"Impersonation Precision: {p[2]:.4f} | Recall: {r[2]:.4f} | F1: {f1[2]:.4f}")

if __name__ == "__main__":
    evaluate_visual_only()
