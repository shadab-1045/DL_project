import os, shutil, json
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.model_c import AntiImpersonationModel
from src.data.model_c_dataset import AntiImpersonationDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

source_dir = 'experiments/model_c_control_C'
target_dir = 'experiments/model_c_control'

shutil.copy(os.path.join(source_dir, 'best_model.pt'), os.path.join(target_dir, 'best_model.pt'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AntiImpersonationModel().to(device)
model.load_state_dict(torch.load(os.path.join(target_dir, 'best_model.pt'), map_location=device, weights_only=True))
model.eval()

val_ds = AntiImpersonationDataset('data/manifests/val_pairs.csv')
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, embs, labels in tqdm(val_loader, desc="Eval"):
        logits = model(imgs.to(device), embs.to(device))
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Genuine', 'Different', 'Impersonation'], yticklabels=['Genuine', 'Different', 'Impersonation'])
plt.ylabel('True')
plt.xlabel('Predicted')
plt.title('Validation Confusion Matrix (Config C)')
plt.savefig(os.path.join(target_dir, 'confusion_matrix.png'))
plt.close()

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
acc = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1, 2])

val_metrics = {
    'accuracy': acc,
    'precision': precision.tolist(),
    'recall': recall.tolist(),
    'f1': f1.tolist(),
    'class_0_support': all_labels.count(0),
    'class_1_support': all_labels.count(1),
    'class_2_support': all_labels.count(2),
    'macro_f1': float(sum(f1)/3)
}
with open(os.path.join(target_dir, 'val_metrics.json'), 'w') as f:
    json.dump(val_metrics, f, indent=4)
print("Finished setting up best model.")
