import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from tqdm import tqdm
import json
import hashlib
import copy

from src.models.model_c_v4 import ModelCV4
from src.data.model_c_v4_dataset import AntiImpersonationV4Dataset

def calculate_class_weights(dataset):
    counts = [0, 0, 0]
    for pair in dataset.pairs:
        counts[dataset.class_map[pair['label']]] += 1
    return counts

def run_integrity_tests(model, dummy_img, dummy_sim):
    print("Running Integrity Tests...")
    # 1. Parameter counts
    vis_params = sum(p.numel() for p in model.visual_branch.parameters())
    id_params = sum(p.numel() for p in model.identity_branch.parameters())
    fus_params = sum(p.numel() for p in model.fusion.parameters())
    print(f"  Visual Branch Params: {vis_params}")
    print(f"  Identity Branch Params: {id_params}")
    print(f"  Fusion Branch Params: {fus_params}")
    
    # 2. Forward pass dimensional constraints
    logits, p_synth, p_id = model(dummy_img, dummy_sim)
    assert p_synth.shape[1] == 1, f"Expected 1D p_synth, got {p_synth.shape}"
    assert p_id.shape[1] == 1, f"Expected 1D p_id, got {p_id.shape}"
    assert logits.shape[1] == 3, f"Expected 3D fusion out, got {logits.shape}"
    
    # 3. Frozen branch gradient constraints
    for p in model.visual_branch.parameters():
        p.requires_grad = False
    for p in model.identity_branch.parameters():
        p.requires_grad = False
        
    opt_fus = optim.Adam(model.fusion.parameters(), lr=1e-3)
    opt_fus.zero_grad()
    loss = logits.sum()
    loss.backward()
    
    has_vis_grad = any(p.grad is not None for p in model.visual_branch.parameters())
    has_id_grad = any(p.grad is not None for p in model.identity_branch.parameters())
    has_fus_grad = any(p.grad is not None for p in model.fusion.parameters())
    
    assert not has_vis_grad, "Visual branch received gradient during fusion!"
    assert not has_id_grad, "Identity branch received gradient during fusion!"
    assert has_fus_grad, "Fusion branch did not receive gradient!"
    print("  [PASS] All Integrity Tests Passed.\n")

def train_fusion():
    torch.manual_seed(42)
    np.random.seed(42)
    
    exp_dir = 'experiments/model_c_v4'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_ds = AntiImpersonationV4Dataset('data/manifests/train_pairs_v2.csv')
    val_ds = AntiImpersonationV4Dataset('data/manifests/val_pairs_v2.csv')
    
    counts = calculate_class_weights(train_ds)
    sample_weights = [0.0] * len(train_ds)
    weight_per_class = [1.0/counts[i] for i in range(3)]
    for idx, pair in enumerate(train_ds.pairs):
        sample_weights[idx] = weight_per_class[train_ds.class_map[pair['label']]]
        
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    model = ModelCV4().to(device)
    model.load_state_dict(torch.load('experiments/model_c_v4/best_model_id_fixed.pt', map_location=device))
    
    # Snapshot parameters to verify they don't change
    orig_vis_state = copy.deepcopy(model.visual_branch.state_dict())
    orig_id_state = copy.deepcopy(model.identity_branch.state_dict())
    
    model.eval() # Ensure batchnorm doesn't update during integrity test
    dummy_img = torch.randn(2, 3, 224, 224).to(device)
    dummy_sim = torch.randn(2, 1).to(device)
    run_integrity_tests(model, dummy_img, dummy_sim)
    
    # Pre-extract Train Features
    print("Extracting training features...")
    train_ps, train_pi, train_labels = [], [], []
    # Use sequential loader for extraction to match indices for weights
    train_extract_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=0)
    with torch.no_grad():
        for imgs, sims, fusion_label, _, _ in tqdm(train_extract_loader, desc="Train Extract"):
            imgs, sims = imgs.to(device), sims.to(device)
            _, p_synth, p_id = model(imgs, sims)
            train_ps.extend(p_synth.cpu().numpy())
            train_pi.extend(p_id.cpu().numpy())
            train_labels.extend(fusion_label.numpy())
            
    train_ps = torch.tensor(train_ps, dtype=torch.float32).to(device)
    train_pi = torch.tensor(train_pi, dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
    
    # Pre-extract Val Features
    print("Extracting validation features...")
    val_ps, val_pi, val_labels = [], [], []
    with torch.no_grad():
        for imgs, sims, fusion_label, _, _ in tqdm(val_loader, desc="Val Extract"):
            imgs, sims = imgs.to(device), sims.to(device)
            _, p_synth, p_id = model(imgs, sims)
            val_ps.extend(p_synth.cpu().numpy())
            val_pi.extend(p_id.cpu().numpy())
            val_labels.extend(fusion_label.numpy())
            
    val_ps = torch.tensor(val_ps, dtype=torch.float32).to(device)
    val_pi = torch.tensor(val_pi, dtype=torch.float32).to(device)
    val_labels = torch.tensor(val_labels, dtype=torch.long).to(device)
    
    # Freeze branches
    for p in model.visual_branch.parameters():
        p.requires_grad = False
    for p in model.identity_branch.parameters():
        p.requires_grad = False
        
    opt_fus = optim.Adam(model.fusion.parameters(), lr=1e-3)
    crit_ce = nn.CrossEntropyLoss()
    
    # Simple tensordataset for fusion training
    from torch.utils.data import TensorDataset
    fusion_train_ds = TensorDataset(train_ps, train_pi, train_labels)
    # Re-create sampler for the TensorDataset
    fusion_sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    fusion_train_loader = DataLoader(fusion_train_ds, batch_size=32, sampler=fusion_sampler)
    
    best_macro = 0
    epochs = 100 # We can do 100 easily since it's 2D
    print("--- Train Late Fusion ---")
    for epoch in range(epochs):
        model.fusion.train()
        for ps, pi, flabels in fusion_train_loader:
            opt_fus.zero_grad()
            logits = model.fusion(ps, pi)
            loss = crit_ce(logits, flabels)
            loss.backward()
            opt_fus.step()
            
        # Eval Fusion
        model.fusion.eval()
        with torch.no_grad():
            logits = model.fusion(val_ps, val_pi)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_labels = val_labels.cpu().numpy()
                
        acc = accuracy_score(all_labels, preds)
        _, r, f1, _ = precision_recall_fscore_support(all_labels, preds, average=None, labels=[0, 1, 2], zero_division=0)
        macro = np.mean(f1)
        if (epoch+1) % 10 == 0:
            print(f"Ep {epoch+1} Val | Acc: {acc:.4f} | GenR: {r[0]:.4f} | DiffR: {r[1]:.4f} | ImpR: {r[2]:.4f} | MacroF1: {macro:.4f}")
        
        if macro > best_macro:
            best_macro = macro
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_fusion_model.pt'))
            
    # Save checkpoint hash
    with open(os.path.join(exp_dir, 'best_fusion_model.pt'), 'rb') as f:
        ckpt_hash = hashlib.sha256(f.read()).hexdigest()
    with open(os.path.join(exp_dir, 'fusion_checkpoint_hash.txt'), 'w') as f:
        f.write(ckpt_hash)
        
    print("\n--- Final Integrity Check ---")
    model.load_state_dict(torch.load('experiments/model_c_v4/best_fusion_model.pt', map_location=device))
    for k, v in model.visual_branch.state_dict().items():
        assert torch.equal(v, orig_vis_state[k]), f"Visual parameter {k} changed during training!"
    for k, v in model.identity_branch.state_dict().items():
        assert torch.equal(v, orig_id_state[k]), f"Identity parameter {k} changed during training!"
    print("[PASS] Visual and Identity parameters remained strictly untouched.")
    
    # ---------------------------------------------------------
    # Evaluation and Ablations
    # ---------------------------------------------------------
    print("\n--- Final Evaluation and Ablations ---")
    model.eval()
    
    results = {}
    
    # Base
    fus_preds, fus_labels = [], []
    actual_classes = []
    all_imgs, all_sims, all_p_synth, all_p_id = [], [], [], []
    
    with torch.no_grad():
        for imgs, sims, fusion_label, vis_target, id_target in tqdm(val_loader, desc="Extracting eval features"):
            imgs, sims = imgs.to(device), sims.to(device)
            logits, p_synth, p_id = model(imgs, sims)
            fus_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            fus_labels.extend(fusion_label.numpy())
            actual_classes.extend(fusion_label.numpy())
            
            all_imgs.append(imgs.cpu())
            all_sims.append(sims.cpu())
            all_p_synth.append(p_synth.cpu())
            all_p_id.append(p_id.cpu())
            
    # Per-class branch diagnostics
    per_class_diag = {0: {'synth_probs': [], 'id_probs': []}, 
                      1: {'synth_probs': [], 'id_probs': []}, 
                      2: {'synth_probs': [], 'id_probs': []}}
                      
    flat_p_synth = torch.cat(all_p_synth).numpy().flatten()
    flat_p_id = torch.cat(all_p_id).numpy().flatten()
                      
    for i in range(len(actual_classes)):
        c = actual_classes[i]
        per_class_diag[c]['synth_probs'].append(float(flat_p_synth[i]))
        per_class_diag[c]['id_probs'].append(float(flat_p_id[i]))
        
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
    
    print("\nBranch Diagnostic Table:")
    print("Class            | Mean P(Id Match) | Mean P(Synthetic)")
    print("-----------------|------------------|------------------")
    for name, c in [("Genuine", 0), ("Different Person", 1), ("Impersonation", 2)]:
        print(f"{name:<16} | {np.mean(per_class_diag[c]['id_probs']):.4f}           | {np.mean(per_class_diag[c]['synth_probs']):.4f}")
    
    # 2D Late Fusion
    acc = accuracy_score(fus_labels, fus_preds)
    p, r, f1, _ = precision_recall_fscore_support(fus_labels, fus_preds, average=None, labels=[0, 1, 2], zero_division=0)
    
    results['late_fusion'] = {
        'accuracy': float(acc),
        'genuine_recall': float(r[0]),
        'different_recall': float(r[1]),
        'impersonation_recall': float(r[2]),
        'impersonation_precision': float(p[2])
    }
    cm = confusion_matrix(fus_labels, fus_preds, labels=[0, 1, 2])
    results['confusion_matrix'] = cm.tolist()
    
    print("\n--- 2D Late Fusion Performance ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Genuine Recall: {r[0]:.4f}")
    print(f"Different Person Recall: {r[1]:.4f}")
    print(f"Impersonation Recall: {r[2]:.4f} (Precision: {p[2]:.4f})")
    print(f"Confusion Matrix:\n{cm}")
    
    # Ablations
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
    print(f"\n--- Ablation: Neutralized Visual (P(Synth) = {mean_p_synth.item():.4f}) ---")
    print(f"Genuine Recall: {r_abl_v[0]:.4f}")
    print(f"Different Person Recall: {r_abl_v[1]:.4f}")
    print(f"Impersonation Recall: {r_abl_v[2]:.4f}")
    
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
    print(f"\n--- Ablation: Neutralized Identity (P(Id) = {mean_p_id.item():.4f}) ---")
    print(f"Genuine Recall: {r_abl_i[0]:.4f}")
    print(f"Different Person Recall: {r_abl_i[1]:.4f}")
    print(f"Impersonation Recall: {r_abl_i[2]:.4f}")
    
    with open('experiments/model_c_v4/final_fusion_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\nV4 Fusion Pipeline Completed.")

if __name__ == "__main__":
    train_fusion()
