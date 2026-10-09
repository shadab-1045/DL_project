import os
import json
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm
import hashlib

from src.models.model_c_v3 import AntiImpersonationModelV3
from src.data.model_c_v3_dataset import AntiImpersonationV3Dataset
# Using a custom FGSM attack for V3 since inputs changed
# We will define it inline here or in adversarial_v3.py
# Let's put it inline for V3 to avoid clutter

def fgsm_attack_v3(model, imgs, ref_embs, probe_embs, sims, labels, criterion, epsilon):
    """
    FGSM attack for V3. Only perturbs the visual input (imgs).
    """
    imgs_adv = imgs.clone().detach().requires_grad_(True)
    
    logits = model(imgs_adv, ref_embs, probe_embs, sims)
    loss = criterion(logits, labels)
    
    model.zero_grad()
    loss.backward()
    
    data_grad = imgs_adv.grad.data
    sign_data_grad = data_grad.sign()
    
    perturbed_imgs = imgs_adv + epsilon * sign_data_grad
    # ImageNet normalization bounds approximately [-2.11, 2.64] for RGB
    # We won't strictly clamp to [0,1] here as the standard input is normalized, 
    # but we can follow V1's implementation.
    
    return perturbed_imgs.detach()

def get_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def calculate_class_weights(dataset):
    counts = [0, 0, 0]
    for pair in dataset.pairs:
        counts[dataset.label_map[pair['label']]] += 1
    return counts

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    exp_dir = 'experiments/model_c_adv_v3'
    os.makedirs(exp_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    epsilon = 0.05
    alpha = 0.5
    epochs = 15
    lr = 1e-4
    batch_size = 32
    
    train_ds = AntiImpersonationV3Dataset('data/manifests/train_pairs_v2.csv')
    val_ds = AntiImpersonationV3Dataset('data/manifests/val_pairs_v2.csv')
    
    counts = calculate_class_weights(train_ds)
    sample_weights = [0.0] * len(train_ds)
    weight_per_class = [1.0/counts[i] for i in range(3)]
    for idx, pair in enumerate(train_ds.pairs):
        sample_weights[idx] = weight_per_class[train_ds.label_map[pair['label']]]
        
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    
    # We set num_workers=0 because FastIdentityEngine (ONNX) is not fork-safe
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = AntiImpersonationModelV3().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_macro_f1 = 0
    best_epoch = -1
    
    history_file = os.path.join(exp_dir, 'train_history.csv')
    with open(history_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_clean_loss', 'train_adv_loss', 'train_total_loss',
            'val_clean_loss', 'val_accuracy', 'val_macro_f1',
            'genuine_recall', 'different_recall', 'impersonation_recall'
        ])
    
    for epoch in range(epochs):
        model.train()
        train_clean_loss_sum = 0
        train_adv_loss_sum = 0
        train_total_loss_sum = 0
        
        for imgs, ref_embs, probe_embs, sims, labels in tqdm(train_loader, desc=f"Ep {epoch+1} Train"):
            imgs, ref_embs, probe_embs, sims, labels = imgs.to(device), ref_embs.to(device), probe_embs.to(device), sims.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Clean
            logits_clean = model(imgs, ref_embs, probe_embs, sims)
            clean_loss = criterion(logits_clean, labels)
            
            # Adv
            perturbed_imgs = fgsm_attack_v3(model, imgs, ref_embs, probe_embs, sims, labels, criterion, epsilon)
            
            model.train()
            logits_adv = model(perturbed_imgs, ref_embs, probe_embs, sims)
            adv_loss = criterion(logits_adv, labels)
            
            total_loss = alpha * clean_loss + (1.0 - alpha) * adv_loss
            
            total_loss.backward()
            optimizer.step()
            
            train_clean_loss_sum += clean_loss.item()
            train_adv_loss_sum += adv_loss.item()
            train_total_loss_sum += total_loss.item()
            
        avg_clean_loss = train_clean_loss_sum / len(train_loader)
        avg_adv_loss = train_adv_loss_sum / len(train_loader)
        avg_total_loss = train_total_loss_sum / len(train_loader)
        
        # Val
        model.eval()
        val_clean_loss_sum = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, ref_embs, probe_embs, sims, labels in tqdm(val_loader, desc=f"Ep {epoch+1} Val"):
                imgs, ref_embs, probe_embs, sims, labels = imgs.to(device), ref_embs.to(device), probe_embs.to(device), sims.to(device), labels.to(device)
                
                logits = model(imgs, ref_embs, probe_embs, sims)
                loss = criterion(logits, labels)
                val_clean_loss_sum += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        avg_val_clean_loss = val_clean_loss_sum / len(val_loader)
        acc = accuracy_score(all_labels, all_preds)
        _, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1, 2], zero_division=0)
        macro_f1 = np.mean(f1)
        
        with open(history_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1, avg_clean_loss, avg_adv_loss, avg_total_loss,
                avg_val_clean_loss, acc, macro_f1,
                recall[0], recall[1], recall[2]
            ])
            
        print(f"Epoch {epoch+1}: TotLoss: {avg_total_loss:.4f} | ValAcc: {acc:.4f} | ValMacroF1: {macro_f1:.4f} | ImpRec: {recall[2]:.4f}")
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pt'))
            
    metadata = {
        'checkpoint_path': 'experiments/model_c_adv_v3/best_model.pt',
        'selected_epoch': best_epoch,
        'architecture_identifier': 'Model C V3 (EfficientNet-B0 + Explicit Pretrained ArcFace Probes)',
        'epsilon': epsilon,
        'alpha': alpha,
        'batch_size': batch_size
    }
    with open(os.path.join(exp_dir, 'checkpoint_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"V3 Training complete. Best Macro F1: {best_macro_f1:.4f} at epoch {best_epoch}")

if __name__ == "__main__":
    main()
