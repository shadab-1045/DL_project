import os
import csv
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from src.models.model_visual_only import VisualOnlyModel

class VisualOnlyDataset(Dataset):
    def __init__(self, pairs_csv):
        with open(pairs_csv, 'r') as f:
            reader = csv.DictReader(f)
            self.pairs = list(reader)
            
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.label_map = {"genuine": 0, "different_person": 1, "impersonation": 2}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        image_pil = Image.open(pair['probe_path']).convert('RGB')
        image_tensor = self.transform(image_pil)
        label = self.label_map[pair['label']]
        return image_tensor, label

def calculate_class_weights(dataset):
    counts = [0, 0, 0]
    for pair in dataset.pairs:
        counts[dataset.label_map[pair['label']]] += 1
    return counts

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    exp_dir = 'experiments/model_visual_only'
    os.makedirs(exp_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    epochs = 15
    lr = 1e-4
    batch_size = 32
    
    train_ds = VisualOnlyDataset('data/manifests/train_pairs_v2.csv')
    val_ds = VisualOnlyDataset('data/manifests/val_pairs_v2.csv')
    
    counts = calculate_class_weights(train_ds)
    sample_weights = [0.0] * len(train_ds)
    weight_per_class = [1.0/counts[i] for i in range(3)]
    for idx, pair in enumerate(train_ds.pairs):
        sample_weights[idx] = weight_per_class[train_ds.label_map[pair['label']]]
        
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = VisualOnlyModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_macro_f1 = 0
    best_epoch = -1
    
    history_file = os.path.join(exp_dir, 'train_history.csv')
    with open(history_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_loss', 'val_loss', 'val_accuracy', 'val_macro_f1',
            'genuine_recall', 'different_recall', 'impersonation_recall'
        ])
    
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0
        
        for imgs, labels in tqdm(train_loader, desc=f"Ep {epoch+1} Train"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            logits = model(imgs)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            
        avg_train_loss = train_loss_sum / len(train_loader)
        
        # Val
        model.eval()
        val_loss_sum = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Ep {epoch+1} Val"):
                imgs, labels = imgs.to(device), labels.to(device)
                
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        avg_val_loss = val_loss_sum / len(val_loader)
        acc = accuracy_score(all_labels, all_preds)
        _, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1, 2], zero_division=0)
        macro_f1 = np.mean(f1)
        
        with open(history_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1, avg_train_loss, avg_val_loss, acc, macro_f1,
                recall[0], recall[1], recall[2]
            ])
            
        print(f"Epoch {epoch+1}: TrainLoss: {avg_train_loss:.4f} | ValLoss: {avg_val_loss:.4f} | ValAcc: {acc:.4f} | ValMacroF1: {macro_f1:.4f} | ImpRec: {recall[2]:.4f}")
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pt'))
            
    print(f"Visual-Only Training complete. Best Macro F1: {best_macro_f1:.4f} at epoch {best_epoch}")

if __name__ == "__main__":
    main()
