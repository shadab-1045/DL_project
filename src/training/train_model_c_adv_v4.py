import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
import hashlib

from src.models.model_c_v4 import ModelCV4
from src.data.model_c_v4_dataset import AntiImpersonationV4Dataset

def calculate_class_weights(dataset):
    counts = [0, 0, 0]
    for pair in dataset.pairs:
        counts[dataset.class_map[pair['label']]] += 1
    return counts

def eval_visual_branch(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, _, _, vis_target, _ in loader:
            imgs, vis_target = imgs.to(device), vis_target.to(device)
            logit = model.visual_branch(imgs)
            prob = torch.sigmoid(logit).squeeze()
            if prob.dim() == 0:
                prob = prob.unsqueeze(0)
            pred = (prob > 0.5).float()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(vis_target.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    _, r, _, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1], zero_division=0)
    return acc, r[0], r[1]  # r[0] is Real, r[1] is Synthetic

def eval_identity_branch(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for _, sims, _, _, id_target in loader:
            sims, id_target = sims.to(device), id_target.to(device)
            logit = model.identity_branch(sims)
            prob = torch.sigmoid(logit).squeeze()
            if prob.dim() == 0:
                prob = prob.unsqueeze(0)
            pred = (prob > 0.5).float()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(id_target.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    _, r, _, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1], zero_division=0)
    return acc, r[0], r[1]

def run_v4_pipeline():
    torch.manual_seed(42)
    np.random.seed(42)
    
    exp_dir = 'experiments/model_c_v4'
    os.makedirs(exp_dir, exist_ok=True)
    
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
    
    # Num workers 0 because of ArcFace ONNX execution
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    model = ModelCV4().to(device)
    
    # --- STEP 1: Train Visual Branch ---
    print("\n--- STEP 1: Train Visual Branch (Real vs Synthetic) ---")
    opt_vis = optim.Adam(model.visual_branch.parameters(), lr=1e-4)
    crit_bce = nn.BCEWithLogitsLoss()
    
    for epoch in range(10):  # 10 epochs for binary pretraining
        model.visual_branch.train()
        for imgs, _, _, vis_target, _ in tqdm(train_loader, desc=f"Vis Ep {epoch+1}"):
            imgs, vis_target = imgs.to(device), vis_target.to(device)
            opt_vis.zero_grad()
            logits = model.visual_branch(imgs)
            loss = crit_bce(logits.squeeze(), vis_target.float())
            loss.backward()
            opt_vis.step()
            
        v_acc, v_real_r, v_syn_r = eval_visual_branch(model, val_loader, device)
        print(f"Vis Ep {epoch+1} Val | Acc: {v_acc:.4f} | Real Rec: {v_real_r:.4f} | Synth Rec: {v_syn_r:.4f}")
        
    # --- STEP 2: Train Identity Branch ---
    print("\n--- STEP 2: Train Identity Branch (Match vs Non-Match) ---")
    opt_id = optim.Adam(model.identity_branch.parameters(), lr=1e-3)
    
    for epoch in range(5):
        model.identity_branch.train()
        for _, sims, _, _, id_target in tqdm(train_loader, desc=f"Id Ep {epoch+1}"):
            sims, id_target = sims.to(device), id_target.to(device)
            opt_id.zero_grad()
            logits = model.identity_branch(sims)
            loss = crit_bce(logits.squeeze(), id_target.float())
            loss.backward()
            opt_id.step()
            
        i_acc, i_nm_r, i_m_r = eval_identity_branch(model, val_loader, device)
        print(f"Id Ep {epoch+1} Val | Acc: {i_acc:.4f} | Non-Match Rec: {i_nm_r:.4f} | Match Rec: {i_m_r:.4f}")
        
    # --- STEP 3: Train Late Fusion ---
    print("\n--- STEP 3: Train Late Fusion (Frozen Branches) ---")
    for p in model.visual_branch.parameters():
        p.requires_grad = False
    for p in model.identity_branch.parameters():
        p.requires_grad = False
        
    opt_fus = optim.Adam(model.fusion.parameters(), lr=1e-3)
    crit_ce = nn.CrossEntropyLoss()
    
    best_macro = 0
    for epoch in range(5):
        model.train() # fusion is trainable, others freeze
        for imgs, sims, fusion_label, _, _ in tqdm(train_loader, desc=f"Fus Ep {epoch+1}"):
            imgs, sims, fusion_label = imgs.to(device), sims.to(device), fusion_label.to(device)
            opt_fus.zero_grad()
            
            logits, _, _ = model(imgs, sims)
            loss = crit_ce(logits, fusion_label)
            loss.backward()
            opt_fus.step()
            
        # Eval Fusion
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, sims, fusion_label, _, _ in val_loader:
                imgs, sims = imgs.to(device), sims.to(device)
                logits, _, _ = model(imgs, sims)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(fusion_label.numpy())
                
        acc = accuracy_score(all_labels, all_preds)
        _, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1, 2], zero_division=0)
        macro = np.mean(f1)
        print(f"Fus Ep {epoch+1} Val | Acc: {acc:.4f} | GenR: {r[0]:.4f} | DiffR: {r[1]:.4f} | ImpR: {r[2]:.4f}")
        
        if macro > best_macro:
            best_macro = macro
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pt'))
            
    # Save checkpoint hash
    with open(os.path.join(exp_dir, 'best_model.pt'), 'rb') as f:
        ckpt_hash = hashlib.sha256(f.read()).hexdigest()
    with open(os.path.join(exp_dir, 'checkpoint_hash.txt'), 'w') as f:
        f.write(ckpt_hash)
        
    print("V4 Pipeline Completed.")

if __name__ == "__main__":
    run_v4_pipeline()
