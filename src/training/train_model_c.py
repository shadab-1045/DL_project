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

def get_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def calculate_class_weights(dataset):
    counts = [0, 0, 0]
    for _, _, label in dataset:
        counts[label] += 1
    total = sum(counts)
    return counts

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    exp_dir = 'experiments/model_c_control'
    os.makedirs(exp_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_ds = AntiImpersonationDataset('data/manifests/train_pairs.csv')
    val_ds = AntiImpersonationDataset('data/manifests/val_pairs.csv')
    
    counts = calculate_class_weights(train_ds)
    sample_weights = [0.0] * len(train_ds)
    weight_per_class = [1.0/counts[i] for i in range(3)]
    for idx, (_, _, label) in enumerate(train_ds):
        sample_weights[idx] = weight_per_class[label]
        
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    model = AntiImpersonationModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 15
    best_macro_f1 = 0
    best_epoch = -1
    
    history_file = os.path.join(exp_dir, 'train_history.csv')
    with open(history_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_loss', 'val_loss', 'val_accuracy', 'val_macro_f1',
            'genuine_precision', 'genuine_recall', 'genuine_f1',
            'different_precision', 'different_recall', 'different_f1',
            'impersonation_precision', 'impersonation_recall', 'impersonation_f1'
        ])
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, embs, labels in tqdm(train_loader, desc=f"Ep {epoch+1} Train", leave=False):
            optimizer.zero_grad()
            logits = model(imgs.to(device), embs.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, embs, labels in tqdm(val_loader, desc=f"Ep {epoch+1} Val", leave=False):
                logits = model(imgs.to(device), embs.to(device))
                loss = criterion(logits, labels.to(device))
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_loader)
        
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1, 2])
        macro_f1 = np.mean(f1)
        
        with open(history_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1, train_loss, val_loss, acc, macro_f1,
                precision[0], recall[0], f1[0],
                precision[1], recall[1], f1[1],
                precision[2], recall[2], f1[2]
            ])
            
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f} | Macro F1: {macro_f1:.4f}")
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pt'))
            
    # Save metadata
    metadata = {
        'checkpoint_path': 'experiments/model_c_control/best_model.pt',
        'selected_epoch': best_epoch,
        'random_seed': 42,
        'optimizer': 'Adam',
        'learning_rate': 1e-4,
        'sampler': 'WeightedRandomSampler',
        'loss_function': 'CrossEntropyLoss',
        'architecture_identifier': 'EfficientNet-B0 + ArcFace + Cosine Similarity',
        'training_manifest_hash': get_hash('data/manifests/train_pairs.csv'),
        'validation_manifest_hash': get_hash('data/manifests/val_pairs.csv')
    }
    with open(os.path.join(exp_dir, 'checkpoint_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Training complete. Best Macro F1: {best_macro_f1:.4f} at epoch {best_epoch}")

if __name__ == "__main__":
    main()
