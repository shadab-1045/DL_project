import json
import pandas as pd
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.model_c import AntiImpersonationModel
from src.data.model_c_dataset import AntiImpersonationDataset
from src.training.adversarial import fgsm_attack

def load_model(path, device):
    model = AntiImpersonationModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def get_preds(model, dataloader, device, epsilon):
    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()
    
    for batch in dataloader:
        imgs, embs, labels = batch
        imgs = imgs.to(device)
        embs = embs.to(device)
        labels = labels.to(device)
        
        if epsilon > 0:
            imgs = fgsm_attack(model, imgs, embs, labels, criterion, epsilon)
            
        with torch.no_grad():
            logits = model(imgs, embs)
            preds = torch.argmax(logits, dim=1)
            
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        
    return np.array(all_preds), np.array(all_labels)

def paired_bootstrap_test(preds_a, preds_b, labels, metric_fn, target_class=2, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(labels)
    diffs = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        val_a = metric_fn(labels[idx], preds_a[idx], target_class)
        val_b = metric_fn(labels[idx], preds_b[idx], target_class)
        diffs.append(val_a - val_b)
        
    diffs = np.array(diffs)
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    
    # Randomization test for p-value (paired)
    # H0: the models perform identically on average. We can swap their predictions randomly.
    rng = np.random.RandomState(seed)
    real_diff = metric_fn(labels, preds_a, target_class) - metric_fn(labels, preds_b, target_class)
    
    better_count = 0
    for _ in range(n_bootstraps):
        swap = rng.binomial(1, 0.5, n).astype(bool)
        perm_a = np.where(swap, preds_b, preds_a)
        perm_b = np.where(swap, preds_a, preds_b)
        perm_diff = metric_fn(labels, perm_a, target_class) - metric_fn(labels, perm_b, target_class)
        if abs(perm_diff) >= abs(real_diff):
            better_count += 1
            
    p_value = better_count / n_bootstraps
    return real_diff, ci_lower, ci_upper, p_value

def f1_score_metric(labels, preds, target_class):
    tp = np.sum((preds == target_class) & (labels == target_class))
    fp = np.sum((preds == target_class) & (labels != target_class))
    fn = np.sum((preds != target_class) & (labels == target_class))
    if tp + fp == 0: p = 0
    else: p = tp / (tp + fp)
    if tp + fn == 0: r = 0
    else: r = tp / (tp + fn)
    if p + r == 0: return 0
    return 2 * p * r / (p + r)

def recall_metric(labels, preds, target_class):
    tp = np.sum((preds == target_class) & (labels == target_class))
    fn = np.sum((preds != target_class) & (labels == target_class))
    if tp + fn == 0: return 0
    return tp / (tp + fn)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    ds = AntiImpersonationDataset('data/manifests/test_pairs.csv')
    dl = DataLoader(ds, batch_size=32, shuffle=False)
    
    ctrl = load_model('experiments/model_c_control/best_model.pt', device)
    adv = load_model('experiments/model_c_adv/best_model.pt', device)
    
    results = []
    
    for eps in [0.05, 0.10]:
        print(f"Evaluating eps={eps}")
        p_ctrl, labels = get_preds(ctrl, dl, device, eps)
        p_adv, _ = get_preds(adv, dl, device, eps)
        
        # A. Recall eps=0.05
        # B. Recall eps=0.10
        # C. F1 eps=0.05
        # D. F1 eps=0.10
        
        r_diff, r_low, r_high, r_p = paired_bootstrap_test(p_adv, p_ctrl, labels, recall_metric)
        f_diff, f_low, f_high, f_p = paired_bootstrap_test(p_adv, p_ctrl, labels, f1_score_metric)
        
        results.append({
            'metric': 'impersonation_recall',
            'epsilon': eps,
            'difference_c_adv_minus_control': r_diff,
            'ci_lower': r_low,
            'ci_upper': r_high,
            'p_value': r_p,
            'bootstrap_iterations': 1000,
            'test_method': 'paired_randomization'
        })
        
        results.append({
            'metric': 'impersonation_f1',
            'epsilon': eps,
            'difference_c_adv_minus_control': f_diff,
            'ci_lower': f_low,
            'ci_upper': f_high,
            'p_value': f_p,
            'bootstrap_iterations': 1000,
            'test_method': 'paired_randomization'
        })
        
    df = pd.DataFrame(results)
    df.to_csv('experiments/final_evaluation/paired_robustness_statistics.csv', index=False)
    print("Saved paired_robustness_statistics.csv")

if __name__ == "__main__":
    main()
