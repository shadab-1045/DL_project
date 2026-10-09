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

from src.models.model_c import AntiImpersonationModel
from src.data.model_c_dataset import AntiImpersonationDataset
from src.training.adversarial import fgsm_attack

def get_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def sha256_file(path):
    hash_sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def calculate_class_weights(dataset):
    counts = [0, 0, 0]
    for _, _, label in dataset:
        counts[label] += 1
    return counts

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    exp_dir = 'experiments/model_c_adv_v2'
    os.makedirs(exp_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Configuration (Same as V1)
    epsilon = 0.05  
    alpha = 0.5     
    epochs = 15
    lr = 1e-4
    batch_size = 32
    
    train_ds = AntiImpersonationDataset('data/manifests/train_pairs_v2.csv')
    val_ds = AntiImpersonationDataset('data/manifests/val_pairs_v2.csv')
    
    counts = calculate_class_weights(train_ds)
    sample_weights = [0.0] * len(train_ds)
    weight_per_class = [1.0/counts[i] for i in range(3)]
    for idx, (_, _, label) in enumerate(train_ds):
        sample_weights[idx] = weight_per_class[label]
        
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Init Model
    # Explicitly do NOT load v1 weights. This starts from scratch with seed 42.
    model = AntiImpersonationModel().to(device)
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
        
        for imgs, embs, labels in tqdm(train_loader, desc=f"Ep {epoch+1} Train", leave=False):
            imgs, embs, labels = imgs.to(device), embs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass clean
            logits_clean = model(imgs, embs)
            clean_loss = criterion(logits_clean, labels)
            
            # Generate FGSM examples
            perturbed_imgs = fgsm_attack(model, imgs, embs, labels, criterion, epsilon)
            
            # Forward pass adversarial
            model.train() 
            logits_adv = model(perturbed_imgs, embs)
            adv_loss = criterion(logits_adv, labels)
            
            # Combined Loss
            total_loss = alpha * clean_loss + (1.0 - alpha) * adv_loss
            
            total_loss.backward()
            optimizer.step()
            
            train_clean_loss_sum += clean_loss.item()
            train_adv_loss_sum += adv_loss.item()
            train_total_loss_sum += total_loss.item()
            
        avg_clean_loss = train_clean_loss_sum / len(train_loader)
        avg_adv_loss = train_adv_loss_sum / len(train_loader)
        avg_total_loss = train_total_loss_sum / len(train_loader)
        
        # Validation on CLEAN examples only
        model.eval()
        val_clean_loss_sum = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, embs, labels in tqdm(val_loader, desc=f"Ep {epoch+1} Val", leave=False):
                imgs, embs, labels = imgs.to(device), embs.to(device), labels.to(device)
                logits = model(imgs, embs)
                loss = criterion(logits, labels)
                val_clean_loss_sum += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        avg_val_clean_loss = val_clean_loss_sum / len(val_loader)
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1, 2])
        macro_f1 = np.mean(f1)
        
        with open(history_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1, avg_clean_loss, avg_adv_loss, avg_total_loss,
                avg_val_clean_loss, acc, macro_f1,
                recall[0], recall[1], recall[2]
            ])
            
        print(f"Epoch {epoch+1}: Train TotLoss: {avg_total_loss:.4f} | Val Clean Loss: {avg_val_clean_loss:.4f} | Val Acc: {acc:.4f} | Macro F1: {macro_f1:.4f}")
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pt'))
            
    # Save Metadata and Config
    metadata = {
        'checkpoint_path': 'experiments/model_c_adv_v2/best_model.pt',
        'selected_epoch': best_epoch,
        'random_seed': 42,
        'optimizer': 'Adam',
        'learning_rate': lr,
        'sampler': 'WeightedRandomSampler',
        'loss_function': 'alpha*Clean + (1-alpha)*Adversarial',
        'architecture_identifier': 'EfficientNet-B0 + ArcFace + Cosine Similarity',
        'training_manifest_hash': get_hash('data/manifests/train_pairs_v2.csv'),
        'validation_manifest_hash': get_hash('data/manifests/val_pairs_v2.csv'),
        'epsilon': epsilon,
        'alpha': alpha,
        'batch_size': batch_size,
        'checkpoint_sha256': sha256_file(os.path.join(exp_dir, 'best_model.pt'))
    }
    with open(os.path.join(exp_dir, 'checkpoint_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Adversarial Training v2 complete. Best Macro F1: {best_macro_f1:.4f} at epoch {best_epoch}")

if __name__ == "__main__":
    main()
