import os
import csv
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm

from src.models.model_c import AntiImpersonationModel
from src.data.model_c_dataset import AntiImpersonationDataset
from src.training.adversarial import fgsm_attack

def evaluate_model_robustness(model, val_loader, device, criterion, epsilon):
    all_preds, all_labels = [], []
    for imgs, embs, labels in tqdm(val_loader, desc=f"Eval eps={epsilon}", leave=False):
        imgs, embs, labels = imgs.to(device), embs.to(device), labels.to(device)
        
        if epsilon > 0:
            # We must set model to train mode for FGSM to calculate gradients, but wait
            # If model is in eval mode, batchnorm uses running stats. We can still calculate input gradients!
            # Let's ensure input requires_grad
            perturbed_imgs = fgsm_attack(model, imgs, embs, labels, criterion, epsilon)
        else:
            perturbed_imgs = imgs
            
        with torch.no_grad():
            logits = model(perturbed_imgs, embs)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1, 2])
    return acc, precision, recall, f1, support

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    
    val_ds = AntiImpersonationDataset('data/manifests/val_pairs.csv')
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    # Load Models
    c_control = AntiImpersonationModel().to(device)
    c_control.load_state_dict(torch.load('experiments/model_c_control/best_model.pt', map_location=device, weights_only=True))
    c_control.eval()
    
    c_adv = AntiImpersonationModel().to(device)
    c_adv.load_state_dict(torch.load('experiments/model_c_adv/best_model.pt', map_location=device, weights_only=True))
    c_adv.eval()
    
    epsilons = [0.0, 0.01, 0.05, 0.1]
    
    results = []
    
    for eps in epsilons:
        print(f"\nEvaluating C-Control (epsilon={eps})...")
        acc, prec, rec, f1, supp = evaluate_model_robustness(c_control, val_loader, device, criterion, eps)
        results.append({
            'model': 'C-Control',
            'epsilon': eps,
            'accuracy': acc,
            'macro_precision': np.mean(prec),
            'macro_recall': np.mean(rec),
            'macro_f1': np.mean(f1),
            'p_gen': prec[0], 'r_gen': rec[0], 'f_gen': f1[0],
            'p_diff': prec[1], 'r_diff': rec[1], 'f_diff': f1[1],
            'p_imp': prec[2], 'r_imp': rec[2], 'f_imp': f1[2],
            's_gen': supp[0], 's_diff': supp[1], 's_imp': supp[2]
        })
        
        print(f"Evaluating C-Adv (epsilon={eps})...")
        acc, prec, rec, f1, supp = evaluate_model_robustness(c_adv, val_loader, device, criterion, eps)
        results.append({
            'model': 'C-Adv',
            'epsilon': eps,
            'accuracy': acc,
            'macro_precision': np.mean(prec),
            'macro_recall': np.mean(rec),
            'macro_f1': np.mean(f1),
            'p_gen': prec[0], 'r_gen': rec[0], 'f_gen': f1[0],
            'p_diff': prec[1], 'r_diff': rec[1], 'f_diff': f1[1],
            'p_imp': prec[2], 'r_imp': rec[2], 'f_imp': f1[2],
            's_gen': supp[0], 's_diff': supp[1], 's_imp': supp[2]
        })
        
    # Write clean_vs_control.csv (only eps = 0)
    with open('experiments/model_c_adv/clean_vs_control.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'epsilon', 'accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'genuine_f1', 'different_person_f1', 'impersonation_f1'])
        for r in results:
            if r['epsilon'] == 0.0:
                writer.writerow([
                    r['model'], r['epsilon'], r['accuracy'], r['macro_precision'], r['macro_recall'], r['macro_f1'],
                    r['f_gen'], r['f_diff'], r['f_imp']
                ])
                
    # Write robustness_report.csv
    with open('experiments/model_c_adv/robustness_report.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'epsilon', 'class', 'precision', 'recall', 'f1', 'support'])
        classes = ['genuine', 'different_person', 'impersonation']
        prefixes = ['gen', 'diff', 'imp']
        for r in results:
            for i, cls in enumerate(classes):
                pref = prefixes[i]
                writer.writerow([
                    r['model'], r['epsilon'], cls,
                    r[f'p_{pref}'], r[f'r_{pref}'], r[f'f_{pref}'], r[f's_{pref}']
                ])
                
    print("\nRobustness evaluation complete.")

if __name__ == "__main__":
    main()
