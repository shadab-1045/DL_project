import os
import json
import csv
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.model_c import AntiImpersonationModel
from src.data.model_c_dataset import AntiImpersonationDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def evaluate_model(model, val_loader, device):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, embs, labels in tqdm(val_loader, desc="Eval"):
            logits = model(imgs.to(device), embs.to(device))
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds

def main():
    target_dir = 'experiments/model_c_control'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = AntiImpersonationModel().to(device)
    model.load_state_dict(torch.load(os.path.join(target_dir, 'best_model.pt'), map_location=device, weights_only=True))
    model.eval()

    val_ds = AntiImpersonationDataset('data/manifests/val_pairs.csv')
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    print("Running Run A...")
    labels_A, preds_A = evaluate_model(model, val_loader, device)
    
    print("Running Run B...")
    labels_B, preds_B = evaluate_model(model, val_loader, device)
    
    if preds_A == preds_B:
        print("Reproducibility Test Passed: Predictions are identical across runs.")
    else:
        print("Reproducibility Test Failed: Predictions differ.")
        
    all_labels, all_preds = labels_A, preds_A

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Genuine', 'Different', 'Impersonation'], yticklabels=['Genuine', 'Different', 'Impersonation'])
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Validation Confusion Matrix')
    plt.savefig(os.path.join(target_dir, 'confusion_matrix.png'))
    plt.close()

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1, 2])
    
    val_metrics = {
        'accuracy': acc,
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist(),
        'class_0_support': all_labels.count(0),
        'class_1_support': all_labels.count(1),
        'class_2_support': all_labels.count(2),
        'macro_precision': float(sum(precision)/3),
        'macro_recall': float(sum(recall)/3),
        'macro_f1': float(sum(f1)/3)
    }
    with open(os.path.join(target_dir, 'val_metrics.json'), 'w') as f:
        json.dump(val_metrics, f, indent=4)
        
    with open(os.path.join(target_dir, 'confusion_matrix_raw.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['True \ Predicted', 'Predicted Genuine (0)', 'Predicted Different (1)', 'Predicted Impersonation (2)'])
        writer.writerow(['True Genuine (0)', cm[0][0], cm[0][1], cm[0][2]])
        writer.writerow(['True Different (1)', cm[1][0], cm[1][1], cm[1][2]])
        writer.writerow(['True Impersonation (2)', cm[2][0], cm[2][1], cm[2][2]])
        
    print(f"\nFinal Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {val_metrics['macro_f1']:.4f}")
    print(f"Macro Precision: {val_metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {val_metrics['macro_recall']:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nPer-class (Prec/Rec/F1):")
    print(f"Genuine: {precision[0]:.4f} / {recall[0]:.4f} / {f1[0]:.4f}")
    print(f"Different: {precision[1]:.4f} / {recall[1]:.4f} / {f1[1]:.4f}")
    print(f"Impersonation: {precision[2]:.4f} / {recall[2]:.4f} / {f1[2]:.4f}")

if __name__ == "__main__":
    main()
