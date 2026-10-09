import os
import torch
import hashlib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.model_c_v4 import ModelCV4
from src.data.model_c_v4_dataset import AntiImpersonationV4Dataset

def hash_file(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def run_test_evaluation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Verify Hashes
    expected_v11 = "0c5e8463023e2d603070ad4aec435be2bfc16c14115237942e38380983acd192"
    expected_v12 = "ca9394dbb3e737e518de4aa77632c3e25b532abce5034e25cfe4a5d11c34837c"
    expected_v13 = "f2647012bc2863d988e65f980310cac9b60a0beb8c3e068f38b564f4383be2f5"
    
    hash_v11 = hash_file('experiments/model_c_v4/best_model.pt')
    hash_v12 = hash_file('experiments/model_c_v4/best_model_id_fixed.pt')
    hash_v13 = hash_file('experiments/model_c_v4/best_fusion_model.pt')
    
    print("--- 1. Checkpoint Verification ---")
    assert hash_v11 == expected_v11, f"V11 hash mismatch: {hash_v11}"
    assert hash_v12 == expected_v12, f"V12 hash mismatch: {hash_v12}"
    assert hash_v13 == expected_v13, f"V13 hash mismatch: {hash_v13}"
    print("[PASS] All artifact hashes match perfectly.")
    
    # Load test dataset
    test_ds = AntiImpersonationV4Dataset('data/manifests/test_pairs_v2.csv')
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    
    print("\n--- 2. Dataset Composition ---")
    counts = {0: 0, 1: 0, 2: 0}
    for p in test_ds.pairs:
        counts[test_ds.class_map[p['label']]] += 1
    print(f"Genuine: {counts[0]}")
    print(f"Different Person: {counts[1]}")
    print(f"Impersonation: {counts[2]}")
    print(f"Total: {sum(counts.values())}")
    
    model = ModelCV4().to(device)
    model.load_state_dict(torch.load('experiments/model_c_v4/best_fusion_model.pt', map_location=device))
    model.eval()
    
    def run_inference():
        all_imgs, all_sims, all_p_synth, all_p_id = [], [], [], []
        preds, labels = [], []
        
        with torch.no_grad():
            for imgs, sims, fusion_label, _, _ in tqdm(test_loader, desc="Inference"):
                imgs, sims = imgs.to(device), sims.to(device)
                logits, p_synth, p_id = model(imgs, sims)
                
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                labels.extend(fusion_label.numpy())
                
                all_imgs.append(imgs.cpu())
                all_sims.append(sims.cpu())
                all_p_synth.append(p_synth.cpu())
                all_p_id.append(p_id.cpu())
                
        return preds, labels, all_imgs, all_sims, all_p_synth, all_p_id

    print("\n--- 3. Running First Inference Pass ---")
    preds1, labels1, imgs1, sims1, ps1, pi1 = run_inference()
    
    print("\n--- 4. Running Second Inference Pass (Reproducibility) ---")
    preds2, labels2, imgs2, sims2, ps2, pi2 = run_inference()
    
    assert preds1 == preds2, "Reproducibility mismatch in predictions!"
    print("[PASS] Reproducibility confirmed. Identical predictions across passes.")
    
    acc = accuracy_score(labels1, preds1)
    p, r, f1, _ = precision_recall_fscore_support(labels1, preds1, average=None, labels=[0, 1, 2], zero_division=0)
    
    print("\n--- 5. Final Test Metrics ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro P/R/F1: {np.mean(p):.4f} / {np.mean(r):.4f} / {np.mean(f1):.4f}")
    
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(labels1, preds1, average='weighted', zero_division=0)
    print(f"Weighted F1: {f1_w:.4f}")
    
    cm = confusion_matrix(labels1, preds1, labels=[0, 1, 2])
    print(f"Confusion Matrix (0=Gen, 1=Diff, 2=Imp):\n{cm}")
    print("\nPer-Class (P / R / F1):")
    print(f"  Genuine:          {p[0]:.4f} / {r[0]:.4f} / {f1[0]:.4f}")
    print(f"  Different Person: {p[1]:.4f} / {r[1]:.4f} / {f1[1]:.4f}")
    print(f"  Impersonation:    {p[2]:.4f} / {r[2]:.4f} / {f1[2]:.4f}")
    
    flat_ps = torch.cat(ps1).numpy().flatten()
    flat_pi = torch.cat(pi1).numpy().flatten()
    flat_sims = torch.cat(sims1).numpy().flatten()
    
    # 6. Branch Diagnostics
    diags = {0: {'pi': [], 'ps': []}, 1: {'pi': [], 'ps': []}, 2: {'pi': [], 'ps': []}}
    for i in range(len(labels1)):
        diags[labels1[i]]['pi'].append(flat_pi[i])
        diags[labels1[i]]['ps'].append(flat_ps[i])
        
    print("\n--- 6. Branch Diagnostics ---")
    print("Class            | Mean P(Id) | Std P(Id) | Mean P(Synth) | Std P(Synth)")
    for name, c in [("Genuine", 0), ("Different Person", 1), ("Impersonation", 2)]:
        mpi = np.mean(diags[c]['pi'])
        spi = np.std(diags[c]['pi'])
        mps = np.mean(diags[c]['ps'])
        sps = np.std(diags[c]['ps'])
        print(f"{name:<16} | {mpi:.4f}     | {spi:.4f}    | {mps:.4f}         | {sps:.4f}")
        
    # 7. Feature Neutralization Ablations
    # Validation Neutralization constants:
    VAL_MEAN_P_SYNTH = 0.1294
    VAL_MEAN_P_ID = 0.7117
    
    # Ablate Visual (Constant P(Synth))
    abl_vis_preds = []
    with torch.no_grad():
        for i in range(len(imgs1)):
            ps = torch.full_like(ps1[i], VAL_MEAN_P_SYNTH).to(device)
            pi = pi1[i].to(device)
            logits = model.fusion(ps, pi)
            abl_vis_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
    _, r_abl_v, _, _ = precision_recall_fscore_support(labels1, abl_vis_preds, average=None, labels=[0, 1, 2], zero_division=0)
    print("\n--- 7. Feature Neutralization Ablation ---")
    print(f"Neutralize Visual (P(Synth)={VAL_MEAN_P_SYNTH}):")
    print(f"  GenR: {r_abl_v[0]:.4f} | DiffR: {r_abl_v[1]:.4f} | ImpR: {r_abl_v[2]:.4f}")
    
    # Ablate Identity (Constant P(Id))
    abl_id_preds = []
    with torch.no_grad():
        for i in range(len(imgs1)):
            ps = ps1[i].to(device)
            pi = torch.full_like(pi1[i], VAL_MEAN_P_ID).to(device)
            logits = model.fusion(ps, pi)
            abl_id_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
    _, r_abl_i, _, _ = precision_recall_fscore_support(labels1, abl_id_preds, average=None, labels=[0, 1, 2], zero_division=0)
    print(f"Neutralize Identity (P(Id)={VAL_MEAN_P_ID}):")
    print(f"  GenR: {r_abl_i[0]:.4f} | DiffR: {r_abl_i[1]:.4f} | ImpR: {r_abl_i[2]:.4f}")
    
    # 8. Save CSV Predictions
    print("\n--- 8. Saving Predictions ---")
    df_rows = []
    for i, p in enumerate(test_ds.pairs):
        pred_label = "genuine" if preds1[i] == 0 else "different_person" if preds1[i] == 1 else "impersonation"
        correct = (preds1[i] == labels1[i])
        
        row = {
            'pair_id': p.get('pair_id', f"test_{i}"),
            'reference_identity': p.get('reference_identity', ''),
            'visible_identity': p.get('visible_identity', ''),
            'physical_identity': p.get('physical_identity', ''),
            'true_label': p['label'],
            'raw_cosine': float(flat_sims[i]),
            'p_identity_match': float(flat_pi[i]),
            'p_synthetic': float(flat_ps[i]),
            'predicted_class': pred_label,
            'correct': correct,
            'attack_type': p.get('attack_type', ''),
            'generator': p.get('generator', '')
        }
        df_rows.append(row)
        
    df = pd.DataFrame(df_rows)
    df.to_csv('experiments/model_c_v4/test_predictions.csv', index=False)
    
    pred_hash = hash_file('experiments/model_c_v4/test_predictions.csv')
    print(f"Saved to test_predictions.csv (Hash: {pred_hash})")
    
if __name__ == "__main__":
    run_test_evaluation()
