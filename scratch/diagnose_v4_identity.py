import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
import numpy as np
import json
from tqdm import tqdm

from src.models.model_c_v4 import ModelCV4
from src.data.model_c_v4_dataset import AntiImpersonationV4Dataset

def run_diagnosis():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelCV4().to(device)
    model.load_state_dict(torch.load('experiments/model_c_v4/best_model.pt', map_location=device))
    model.eval()
    
    val_ds = AntiImpersonationV4Dataset('data/manifests/val_pairs_v2.csv')
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # Target counts in val_ds
    target_counts = {0.0: 0, 1.0: 0}
    class_counts = {0: 0, 1: 0, 2: 0}
    for pair in val_ds.pairs:
        l = pair['label']
        id_tgt = val_ds.identity_match_map[l]
        f_tgt = val_ds.class_map[l]
        target_counts[id_tgt] += 1
        class_counts[f_tgt] += 1
        
    print(f"Val target counts (Identity Match): 0.0 (Non-Match): {target_counts[0.0]}, 1.0 (Match): {target_counts[1.0]}")
    print(f"Val class counts: Genuine: {class_counts[0]}, Diff: {class_counts[1]}, Imp: {class_counts[2]}")
    
    raw_cosines = {0: [], 1: [], 2: []}
    p_id_matches = {0: [], 1: [], 2: []}
    p_synthetics = {0: [], 1: [], 2: []}
    
    all_sims = []
    all_id_targets = []
    
    all_fusion_labels = []
    all_fusion_preds = []
    
    with torch.no_grad():
        for imgs, sims, fusion_label, vis_target, id_target in tqdm(val_loader):
            imgs, sims = imgs.to(device), sims.to(device)
            
            logits, p_synth, p_id = model(imgs, sims)
            fusion_preds = torch.argmax(logits, dim=1)
            
            all_fusion_labels.extend(fusion_label.numpy())
            all_fusion_preds.extend(fusion_preds.cpu().numpy())
            
            for i in range(len(fusion_label)):
                cls = fusion_label[i].item()
                raw_cosines[cls].append(sims[i].item())
                p_id_matches[cls].append(p_id[i].item())
                p_synthetics[cls].append(p_synth[i].item())
                
                all_sims.append(sims[i].item())
                all_id_targets.append(id_target[i].item())

    # Offline calibration sanity check
    X = np.array(all_sims).reshape(-1, 1)
    y = np.array(all_id_targets)
    clf = LogisticRegression(random_state=42).fit(X, y)
    offline_probs = clf.predict_proba(X)[:, 1]
    
    offline_p_id_matches = {0: [], 1: [], 2: []}
    for i, cls in enumerate(all_fusion_labels):
        offline_p_id_matches[cls].append(offline_probs[i])
        
    # Stats
    for cls, name in [(0, 'Genuine'), (1, 'Diff Person'), (2, 'Impersonation')]:
        print(f"\n{name}:")
        print(f"  Raw Cosine Mean: {np.mean(raw_cosines[cls]):.4f} | Std: {np.std(raw_cosines[cls]):.4f}")
        print(f"  P(Id_Match) Mean: {np.mean(p_id_matches[cls]):.4f} | Std: {np.std(p_id_matches[cls]):.4f}")
        print(f"  P(Synth) Mean: {np.mean(p_synthetics[cls]):.4f} | Std: {np.std(p_synthetics[cls]):.4f}")
        print(f"  Offline P(Id_Match) Mean: {np.mean(offline_p_id_matches[cls]):.4f}")
        
    # Calibrator parameters
    w = model.identity_branch.calibrate.weight.item()
    b = model.identity_branch.calibrate.bias.item()
    print(f"\nTrained Identity Branch Params: w={w:.4f}, b={b:.4f}")
    print(f"Offline LogReg Params: w={clf.coef_[0][0]:.4f}, b={clf.intercept_[0]:.4f}")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_fusion_labels, all_fusion_preds, labels=[0, 1, 2])
    print(f"\nFusion Confusion Matrix:\n{cm}")
    
if __name__ == "__main__":
    run_diagnosis()
