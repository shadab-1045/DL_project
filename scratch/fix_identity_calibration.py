import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import numpy as np
import json
from tqdm import tqdm

from src.models.model_c_v4 import ModelCV4
from src.data.model_c_v4_dataset import AntiImpersonationV4Dataset

def fix_identity_calibration():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_ds = AntiImpersonationV4Dataset('data/manifests/train_pairs_v2.csv')
    val_ds = AntiImpersonationV4Dataset('data/manifests/val_pairs_v2.csv')
    
    # We use num_workers=0 because of ArcFace ONNX
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    print("Extracting train set features for calibration...")
    train_sims = []
    train_targets = []
    with torch.no_grad():
        for _, sims, _, _, id_target in tqdm(train_loader):
            train_sims.extend(sims.numpy())
            train_targets.extend(id_target.numpy())
            
    X_train = np.array(train_sims).reshape(-1, 1)
    y_train = np.array(train_targets)
    
    print("Fitting Logistic Regression (Penalty=None)...")
    clf = LogisticRegression(penalty=None, random_state=42)
    clf.fit(X_train, y_train)
    
    w = clf.coef_[0][0]
    b = clf.intercept_[0]
    
    print(f"Learned Parameters: w={w:.4f}, b={b:.4f}")
    
    # Assert monotonic
    assert w > 0, f"Calibrator is not monotonic! w={w} should be > 0"
    
    # Train metrics
    train_preds = clf.predict(X_train)
    train_probs = clf.predict_proba(X_train)[:, 1]
    train_acc = accuracy_score(y_train, train_preds)
    train_auc = roc_auc_score(y_train, train_probs)
    print(f"Train Identity-Match Acc: {train_acc:.4f} | AUC: {train_auc:.4f}")
    
    # Validation evaluation
    print("\nExtracting val set features for evaluation...")
    val_sims = []
    val_targets = []
    val_fusion_labels = []
    with torch.no_grad():
        for _, sims, fusion_label, _, id_target in tqdm(val_loader):
            val_sims.extend(sims.numpy())
            val_targets.extend(id_target.numpy())
            val_fusion_labels.extend(fusion_label.numpy())
            
    X_val = np.array(val_sims).reshape(-1, 1)
    y_val = np.array(val_targets)
    
    val_preds = clf.predict(X_val)
    val_probs = clf.predict_proba(X_val)[:, 1]
    
    val_acc = accuracy_score(y_val, val_preds)
    val_auc = roc_auc_score(y_val, val_probs)
    cm = confusion_matrix(y_val, val_preds, labels=[0, 1])
    
    print(f"\nValidation Identity-Match Acc: {val_acc:.4f} | AUC: {val_auc:.4f}")
    print(f"Validation Identity-Match Confusion Matrix (0=NonMatch, 1=Match):\n{cm}")
    
    # Per-class stats
    # 0 = Genuine, 1 = Different, 2 = Impersonation
    stats = {0: {'sims': [], 'probs': []}, 1: {'sims': [], 'probs': []}, 2: {'sims': [], 'probs': []}}
    
    for i in range(len(val_fusion_labels)):
        c = val_fusion_labels[i]
        stats[c]['sims'].append(val_sims[i])
        stats[c]['probs'].append(val_probs[i])
        
    print("\nPer-Class Statistics:")
    for c, name in [(0, 'Genuine'), (1, 'Different Person'), (2, 'Impersonation')]:
        mean_sim = np.mean(stats[c]['sims'])
        std_sim = np.std(stats[c]['sims'])
        mean_p = np.mean(stats[c]['probs'])
        std_p = np.std(stats[c]['probs'])
        print(f"  {name}:")
        print(f"    Raw Cosine Mean: {mean_sim:.4f} | Std: {std_sim:.4f}")
        print(f"    Calibrated P(Match) Mean: {mean_p:.4f} | Std: {std_p:.4f}")
        
    # Inject back into model and test
    model = ModelCV4().to(device)
    model.load_state_dict(torch.load('experiments/model_c_v4/best_model.pt', map_location=device))
    
    # Set weights
    model.identity_branch.calibrate.weight.data = torch.tensor([[w]], dtype=torch.float32).to(device)
    model.identity_branch.calibrate.bias.data = torch.tensor([b], dtype=torch.float32).to(device)
    
    torch.save(model.state_dict(), 'experiments/model_c_v4/best_model_id_fixed.pt')
    print("\nSaved new checkpoint to experiments/model_c_v4/best_model_id_fixed.pt")

    # Sanity Test with Synthetic Values
    print("\nRunning Sanity Test with Synthetic Cosine Values...")
    synthetic_sims = torch.tensor([[0.0], [0.1], [0.3], [0.4], [0.8]], dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        out = model.identity_branch(synthetic_sims)
        out_p = torch.sigmoid(out).cpu().numpy().flatten()
    print(f"  Cos=0.0 -> P(Match)={out_p[0]:.4f}")
    print(f"  Cos=0.1 -> P(Match)={out_p[1]:.4f}")
    print(f"  Cos=0.3 -> P(Match)={out_p[2]:.4f}")
    print(f"  Cos=0.4 -> P(Match)={out_p[3]:.4f}")
    print(f"  Cos=0.8 -> P(Match)={out_p[4]:.4f}")

if __name__ == "__main__":
    fix_identity_calibration()
