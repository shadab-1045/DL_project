import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import json
from tqdm import tqdm

from src.models.model_c_v4 import ModelCV4
from src.data.model_c_v4_dataset import AntiImpersonationV4Dataset

def evaluate_v4():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelCV4().to(device)
    model.load_state_dict(torch.load('experiments/model_c_v4/best_model.pt', map_location=device))
    model.eval()
    
    val_ds = AntiImpersonationV4Dataset('data/manifests/val_pairs_v2.csv')
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    results = {}
    
    # 1. Identity-only (Branch diagnostic)
    id_preds, id_labels = [], []
    id_probs = []
    
    # 2. Visual-only (Branch diagnostic)
    vis_preds, vis_labels = [], []
    vis_probs = []
    
    # 3. 2D Late Fusion (Normal)
    fus_preds, fus_labels = [], []
    
    # Class mappings
    actual_classes = []
    
    # Ablation buffers
    all_imgs = []
    all_sims = []
    all_p_synth = []
    all_p_id = []
    
    with torch.no_grad():
        for imgs, sims, fusion_label, vis_target, id_target in tqdm(val_loader, desc="Extracting features"):
            imgs, sims = imgs.to(device), sims.to(device)
            
            logits, p_synth, p_id = model(imgs, sims)
            
            fus_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            fus_labels.extend(fusion_label.numpy())
            actual_classes.extend(fusion_label.numpy())
            
            vis_pred = (p_synth.squeeze() > 0.5).float()
            if vis_pred.dim() == 0: vis_pred = vis_pred.unsqueeze(0)
            vis_preds.extend(vis_pred.cpu().numpy())
            vis_labels.extend(vis_target.numpy())
            vis_probs.extend(p_synth.squeeze().cpu().numpy().reshape(-1))
            
            id_pred = (p_id.squeeze() > 0.5).float()
            if id_pred.dim() == 0: id_pred = id_pred.unsqueeze(0)
            id_preds.extend(id_pred.cpu().numpy())
            id_labels.extend(id_target.numpy())
            id_probs.extend(p_id.squeeze().cpu().numpy().reshape(-1))
            
            all_imgs.append(imgs.cpu())
            all_sims.append(sims.cpu())
            all_p_synth.append(p_synth.cpu())
            all_p_id.append(p_id.cpu())
            
    # Compile branch diagnostics
    vis_acc = accuracy_score(vis_labels, vis_preds)
    id_acc = accuracy_score(id_labels, id_preds)
    
    results['branch_diagnostics'] = {
        'visual_binary_accuracy': float(vis_acc),
        'identity_binary_accuracy': float(id_acc)
    }
    
    # Per-class branch diagnostics
    per_class_diag = {0: {'synth_probs': [], 'id_probs': []}, 
                      1: {'synth_probs': [], 'id_probs': []}, 
                      2: {'synth_probs': [], 'id_probs': []}}
                      
    for i in range(len(actual_classes)):
        c = actual_classes[i]
        per_class_diag[c]['synth_probs'].append(float(vis_probs[i]))
        per_class_diag[c]['id_probs'].append(float(id_probs[i]))
        
    results['per_class_probabilities'] = {
        'genuine': {
            'mean_p_synthetic': float(np.mean(per_class_diag[0]['synth_probs'])),
            'mean_p_id_match': float(np.mean(per_class_diag[0]['id_probs']))
        },
        'different_person': {
            'mean_p_synthetic': float(np.mean(per_class_diag[1]['synth_probs'])),
            'mean_p_id_match': float(np.mean(per_class_diag[1]['id_probs']))
        },
        'impersonation': {
            'mean_p_synthetic': float(np.mean(per_class_diag[2]['synth_probs'])),
            'mean_p_id_match': float(np.mean(per_class_diag[2]['id_probs']))
        }
    }
    
    # Late Fusion
    acc = accuracy_score(fus_labels, fus_preds)
    p, r, f1, _ = precision_recall_fscore_support(fus_labels, fus_preds, average=None, labels=[0, 1, 2], zero_division=0)
    
    results['late_fusion'] = {
        'accuracy': float(acc),
        'genuine_recall': float(r[0]),
        'different_recall': float(r[1]),
        'impersonation_recall': float(r[2]),
        'impersonation_precision': float(p[2])
    }
    
    # Ablations
    # Mean neutral values
    P_synth_all = torch.cat(all_p_synth)
    P_id_all = torch.cat(all_p_id)
    mean_p_synth = torch.mean(P_synth_all, dim=0, keepdim=True).to(device)
    mean_p_id = torch.mean(P_id_all, dim=0, keepdim=True).to(device)
    
    # Neutralize Visual
    abl_vis_preds = []
    with torch.no_grad():
        for i in range(len(all_imgs)):
            ps = mean_p_synth.expand_as(all_p_synth[i]).to(device)
            pi = all_p_id[i].to(device)
            logits = model.fusion(ps, pi)
            abl_vis_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
    _, r_abl_v, _, _ = precision_recall_fscore_support(fus_labels, abl_vis_preds, average=None, labels=[0, 1, 2], zero_division=0)
    
    results['ablation_neutralize_visual'] = {
        'genuine_recall': float(r_abl_v[0]),
        'different_recall': float(r_abl_v[1]),
        'impersonation_recall': float(r_abl_v[2])
    }
    
    # Neutralize Identity
    abl_id_preds = []
    with torch.no_grad():
        for i in range(len(all_imgs)):
            ps = all_p_synth[i].to(device)
            pi = mean_p_id.expand_as(all_p_id[i]).to(device)
            logits = model.fusion(ps, pi)
            abl_id_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
    _, r_abl_i, _, _ = precision_recall_fscore_support(fus_labels, abl_id_preds, average=None, labels=[0, 1, 2], zero_division=0)
    
    results['ablation_neutralize_identity'] = {
        'genuine_recall': float(r_abl_i[0]),
        'different_recall': float(r_abl_i[1]),
        'impersonation_recall': float(r_abl_i[2])
    }
    
    with open('experiments/model_c_v4/val_robustness.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print(json.dumps(results, indent=4))
    
if __name__ == "__main__":
    evaluate_v4()
