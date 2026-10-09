import os
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
from tqdm import tqdm
import numpy as np

from src.models.model_c import AntiImpersonationModel
from src.data.model_c_dataset import AntiImpersonationDataset

def calculate_class_weights(dataset):
    counts = [0, 0, 0]
    for _, _, label in dataset:
        counts[label] += 1
    total = sum(counts)
    weights = [total / (len(counts) * c) for c in counts]
    return torch.FloatTensor(weights), counts

def run_experiment(config_name, config):
    exp_dir = f"experiments/model_c_control_{config_name}"
    os.makedirs(exp_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_ds = AntiImpersonationDataset('data/manifests/train_pairs.csv')
    val_ds = AntiImpersonationDataset('data/manifests/val_pairs.csv')
    
    class_weights, counts = calculate_class_weights(train_ds)
    
    if config['sampler'] == 'weighted':
        sample_weights = [0] * len(train_ds)
        # Using inverse counts as weights for each sample
        weight_per_class = [1.0/counts[i] for i in range(3)]
        for idx, (_, _, label) in enumerate(train_ds):
            sample_weights[idx] = weight_per_class[label]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
        criterion = nn.CrossEntropyLoss() # standard CE
    else:
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    model = AntiImpersonationModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    best_macro_f1 = 0
    patience = 5
    patience_counter = 0
    
    final_metrics = {}
    
    for epoch in range(config['epochs']):
        model.train()
        for imgs, embs, labels in tqdm(train_loader, desc=f"[{config_name}] Ep {epoch+1} Trn", leave=False):
            optimizer.zero_grad()
            logits = model(imgs.to(device), embs.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, embs, labels in tqdm(val_loader, desc=f"[{config_name}] Ep {epoch+1} Val", leave=False):
                logits = model(imgs.to(device), embs.to(device))
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1, 2])
        macro_f1 = np.mean(f1)
        
        print(f"[{config_name}] Epoch {epoch+1}: Macro F1: {macro_f1:.4f} | Diff Recall: {recall[1]:.4f}")
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pt'))
            patience_counter = 0
            final_metrics = {
                'Config': config_name,
                'Val Accuracy': acc,
                'Macro Precision': np.mean(precision),
                'Macro Recall': np.mean(recall),
                'Macro F1': macro_f1,
                'Genuine Recall': recall[0],
                'Diff Recall': recall[1],
                'Impersonation Recall': recall[2],
                'Genuine F1': f1[0],
                'Diff F1': f1[1],
                'Impersonation F1': f1[2]
            }
        else:
            patience_counter += 1
            if config.get('early_stopping') and patience_counter >= patience:
                print(f"[{config_name}] Early stopping triggered.")
                break
                
    return final_metrics

if __name__ == "__main__":
    configs = {
        'A': {'epochs': 10, 'sampler': 'none', 'early_stopping': False},
        'B': {'epochs': 20, 'sampler': 'none', 'early_stopping': True},
        'C': {'epochs': 15, 'sampler': 'weighted', 'early_stopping': False}
    }
    
    results = []
    for name, cfg in configs.items():
        torch.manual_seed(42)
        np.random.seed(42)
        metrics = run_experiment(name, cfg)
        results.append(metrics)
        
    df = pd.DataFrame(results)
    df.to_csv('experiments/model_c_control/baseline_comparison.csv', index=False)
    print("Baseline comparison completed.")
