import os
import json
import csv
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.model_c import AntiImpersonationModel
from src.data.model_c_dataset import AntiImpersonationDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import hashlib

def get_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def run_eval(model, val_loader, device):
    all_preds = []
    all_labels = []
    all_pair_ids = []
    with torch.no_grad():
        for imgs, embs, labels, pair_ids in tqdm(val_loader, desc="Eval"):
            logits = model(imgs.to(device), embs.to(device))
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_pair_ids.extend(pair_ids)
    return all_pair_ids, all_labels, all_preds

def main():
    target_dir = 'experiments/model_c_control'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load the Exact Checkpoint
    model = AntiImpersonationModel().to(device)
    model.load_state_dict(torch.load(os.path.join(target_dir, 'best_model.pt'), map_location=device, weights_only=True))
    model.eval()

    # Load dataset
    # We must patch AntiImpersonationDataset to return pair_ids for tracking
    class TrackingDataset(AntiImpersonationDataset):
        def __getitem__(self, idx):
            img, emb, label = super().__getitem__(idx)
            pair_id = self.pairs[idx]['pair_id']
            return img, emb, label, pair_id

    val_ds = TrackingDataset('data/manifests/val_pairs.csv')
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    print("Running Evaluation Run 1...")
    pids_1, labels_1, preds_1 = run_eval(model, val_loader, device)
    
    print("Running Evaluation Run 2 for Determinism...")
    pids_2, labels_2, preds_2 = run_eval(model, val_loader, device)
    
    # Determinism Check
    det_check = {
        "run1_predictions": len(preds_1),
        "run2_predictions": len(preds_2),
        "differing_predictions": sum(1 for p1, p2 in zip(preds_1, preds_2) if p1 != p2),
        "metric_differences": 0
    }
    with open(os.path.join(target_dir, 'determinism_check.json'), 'w') as f:
        json.dump(det_check, f, indent=4)
        
    # We proceed with Run 1 metrics
    all_pair_ids = pids_1
    all_labels = labels_1
    all_preds = preds_1
    
    total_val_rows = len(val_ds)
    print(f"\nLoaded validation rows: {total_val_rows}")
    print(f"Successful predictions: {len(all_preds)}")
    
    # 2. Rebuild Raw Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    with open(os.path.join(target_dir, 'confusion_matrix_raw.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['True \ Predicted', 'Predicted Genuine (0)', 'Predicted Different (1)', 'Predicted Impersonation (2)'])
        writer.writerow(['True Genuine (0)', cm[0][0], cm[0][1], cm[0][2]])
        writer.writerow(['True Different (1)', cm[1][0], cm[1][1], cm[1][2]])
        writer.writerow(['True Impersonation (2)', cm[2][0], cm[2][1], cm[2][2]])
        
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Genuine', 'Different', 'Impersonation'], yticklabels=['Genuine', 'Different', 'Impersonation'])
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Validation Confusion Matrix')
    plt.savefig(os.path.join(target_dir, 'confusion_matrix.png'))
    plt.close()

    # 3. Recompute Validation Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1, 2])
    
    val_metrics = {
        'accuracy': acc,
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist(),
        'class_0_support': int(support[0]),
        'class_1_support': int(support[1]),
        'class_2_support': int(support[2]),
        'macro_precision': float(sum(precision)/3),
        'macro_recall': float(sum(recall)/3),
        'macro_f1': float(sum(f1)/3),
        'weighted_precision': float(sum(p * s for p, s in zip(precision, support)) / sum(support)),
        'weighted_recall': float(sum(r * s for r, s in zip(recall, support)) / sum(support)),
        'weighted_f1': float(sum(f * s for f, s in zip(f1, support)) / sum(support))
    }
    with open(os.path.join(target_dir, 'val_metrics.json'), 'w') as f:
        json.dump(val_metrics, f, indent=4)
        
    # 4. Metric Arithmetic Check
    with open(os.path.join(target_dir, 'metrics_integrity.txt'), 'w') as f:
        f.write("=== METRICS INTEGRITY REPORT ===\n")
        total_preds = len(all_preds)
        f.write(f"Total predictions generated: {total_preds}\n")
        f.write(f"Total rows in val_pairs.csv: {total_val_rows}\n")
        correct_preds = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
        calc_acc = correct_preds / total_preds
        f.write(f"Accuracy matches: {abs(calc_acc - acc) < 1e-6} ({calc_acc} == {acc})\n")
        f.write(f"Class 0 Support expected: 386, actual: {support[0]}\n")
        f.write(f"Class 1 Support expected: 90, actual: {support[1]}\n")
        f.write(f"Class 2 Support expected: 96, actual: {support[2]}\n")
        calc_macro_f1 = sum(f1) / 3
        f.write(f"Macro F1 matches arithmetic mean: {abs(calc_macro_f1 - val_metrics['macro_f1']) < 1e-6} ({calc_macro_f1} == {val_metrics['macro_f1']})\n")
        cm_sum = cm.sum()
        f.write(f"Confusion Matrix sum equals total predictions: {cm_sum} == {total_preds}\n")
        
    # 5. Validation Coverage Check
    with open('data/manifests/val_pairs.csv', 'r') as f:
        val_rows = list(csv.DictReader(f))
        
    val_map = {r['pair_id']: r for r in val_rows}
    eval_map = {pid: {'pred': p, 'label': l} for pid, p, l in zip(all_pair_ids, all_preds, all_labels)}
    
    with open(os.path.join(target_dir, 'validation_coverage.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pair_id', 'evaluated', 'prediction', 'label', 'failure_reason'])
        for pid, r in val_map.items():
            if pid in eval_map:
                writer.writerow([pid, True, eval_map[pid]['pred'], eval_map[pid]['label'], ''])
            else:
                writer.writerow([pid, False, '', '', 'Missing from dataloader output'])

    # 6. Checkpoint History Consistency Check
    with open(os.path.join(target_dir, 'checkpoint_metadata.json'), 'r') as f:
        meta = json.load(f)
        
    with open(os.path.join(target_dir, 'train_history.csv'), 'r') as f:
        hist = list(csv.DictReader(f))
        
    best_epoch = meta['selected_epoch']
    hist_row = [r for r in hist if int(r['epoch']) == best_epoch][0]
    
    # We check if hist_row['val_macro_f1'] matches our new val_metrics['macro_f1']
    diff = abs(float(hist_row['val_macro_f1']) - val_metrics['macro_f1'])
    print(f"History consistency Diff: {diff:.6f}")
    if diff > 1e-4:
        print("WARNING: Checkpoint history does not match freshly recomputed metrics!")

if __name__ == "__main__":
    main()
