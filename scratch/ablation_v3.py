import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import collections
from tqdm import tqdm

from src.models.model_c_v3 import AntiImpersonationModelV3
from src.data.model_c_v3_dataset import AntiImpersonationV3Dataset

def compute_metrics(y_true, y_pred, y_probs):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    unique, counts = np.unique(y_pred, return_counts=True)
    pred_counts = dict(zip(unique, counts))
    
    # Calculate probabilities per actual class
    probs_by_class = {0: [], 1: [], 2: []}
    for i in range(len(y_true)):
        probs_by_class[int(y_true[i])].append(y_probs[i])
        
    mean_probs = {}
    for c in [0, 1, 2]:
        if probs_by_class[c]:
            mean_probs[str(c)] = np.mean(probs_by_class[c], axis=0)
        else:
            mean_probs[str(c)] = np.array([0.0, 0.0, 0.0])
            
    # Also fix pred_counts keys
    pred_counts = {str(k): int(v) for k, v in pred_counts.items()}
            
    return {
        "accuracy": acc,
        "macro_f1": np.mean(f1),
        "gen_recall": r[0],
        "diff_recall": r[1],
        "imp_precision": p[2],
        "imp_recall": r[2],
        "imp_f1": f1[2],
        "cm": cm,
        "pred_counts": pred_counts,
        "mean_probs_by_actual": mean_probs
    }

def run_ablation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AntiImpersonationModelV3().to(device)
    model.load_state_dict(torch.load('experiments/model_c_adv_v3/best_model.pt', map_location=device))
    model.eval()
    
    val_ds = AntiImpersonationV3Dataset('data/manifests/val_pairs_v2.csv')
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # Pre-extract all features to compute means and run fast ablations
    all_imgs, all_refs, all_probes, all_sims, all_labels = [], [], [], [], []
    all_vis_feats = []
    
    print("Extracting features for ablation...")
    with torch.no_grad():
        for imgs, refs, probes, sims, labels in tqdm(val_loader):
            imgs, refs, probes, sims = imgs.to(device), refs.to(device), probes.to(device), sims.to(device)
            vis = model.backbone(imgs)
            
            all_vis_feats.append(vis.cpu())
            all_refs.append(refs.cpu())
            all_probes.append(probes.cpu())
            all_sims.append(sims.cpu())
            all_labels.append(labels.cpu())
            
    V = torch.cat(all_vis_feats)
    R = torch.cat(all_refs)
    P = torch.cat(all_probes)
    S = torch.cat(all_sims)
    L = torch.cat(all_labels).numpy()
    
    # Compute neutral representations
    mean_V = torch.mean(V, dim=0, keepdim=True)
    mean_R = torch.mean(R, dim=0, keepdim=True)
    mean_P = torch.mean(P, dim=0, keepdim=True)
    mean_S = torch.mean(S, dim=0, keepdim=True)
    
    # We will pass these directly to the fusion layer
    def evaluate_fusion(v, r, p, s):
        y_pred = []
        y_probs = []
        logits_all = []
        
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(L), batch_size):
                bv = v[i:i+batch_size].to(device)
                br = r[i:i+batch_size].to(device)
                bp = p[i:i+batch_size].to(device)
                bs = s[i:i+batch_size].to(device)
                
                if bs.dim() == 1:
                    bs = bs.unsqueeze(1)
                
                fused = torch.cat([bv, br, bp, bs], dim=1)
                logits = model.fusion(fused)
                probs = torch.softmax(logits, dim=1)
                
                logits_all.append(logits.cpu())
                y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
                y_probs.extend(probs.cpu().numpy())
                
        return np.array(y_pred), np.array(y_probs), torch.cat(logits_all)

    results = {}
    
    # 1. BASELINE
    pred, probs, logits_base = evaluate_fusion(V, R, P, S)
    results['Baseline'] = compute_metrics(L, pred, probs)
    
    # 2. IDENTITY-ONLY (Neutralize V)
    pred, probs, _ = evaluate_fusion(mean_V.expand_as(V), R, P, S)
    results['IdentityOnly'] = compute_metrics(L, pred, probs)
    
    # 3. VISUAL-ONLY (Neutralize R, P, S)
    pred, probs, _ = evaluate_fusion(V, mean_R.expand_as(R), mean_P.expand_as(P), mean_S.expand_as(S))
    results['VisualOnly'] = compute_metrics(L, pred, probs)
    
    # 4. VISUAL + COSINE (Neutralize R, P)
    pred, probs, _ = evaluate_fusion(V, mean_R.expand_as(R), mean_P.expand_as(P), S)
    results['Visual+Cosine'] = compute_metrics(L, pred, probs)
    
    # 5. REFERENCE + PROBE WITHOUT EXPLICIT COSINE (Neutralize S)
    pred, probs, _ = evaluate_fusion(V, R, P, mean_S.expand_as(S))
    results['RefProbeNoCosine'] = compute_metrics(L, pred, probs)
    
    # 7. VISUAL-BRANCH SENSITIVITY
    # Compare logits_base vs logits_id_only (where V is neutral)
    _, _, logits_id_only = evaluate_fusion(mean_V.expand_as(V), R, P, S)
    logit_diff = torch.abs(logits_base - logits_id_only)
    
    sens_mean = torch.mean(logit_diff).item()
    sens_max = torch.max(logit_diff).item()
    
    base_preds = torch.argmax(logits_base, dim=1)
    id_only_preds = torch.argmax(logits_id_only, dim=1)
    changed_pct = torch.mean((base_preds != id_only_preds).float()).item() * 100.0
    
    results['Sensitivity'] = {
        'mean_abs_logit_diff': sens_mean,
        'max_abs_logit_diff': sens_max,
        'changed_pct': changed_pct
    }
    
    import json
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
            
    with open('scratch/ablation_results.json', 'w') as f:
        json.dump(results, f, cls=NumpyEncoder, indent=4)
        
    print("Ablation completed.")

if __name__ == "__main__":
    run_ablation()
