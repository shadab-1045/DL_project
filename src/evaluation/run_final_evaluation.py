import os
import json
import csv
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve
from tqdm import tqdm
import cv2

from src.identity.arcface import IdentityEngine
from src.models.model_c import AntiImpersonationModel
from src.data.model_c_dataset import AntiImpersonationDataset
from torch.utils.data import DataLoader
from src.training.adversarial import fgsm_attack

from insightface.app import FaceAnalysis

class FastIdentityEngine:
    def __init__(self):
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=-1)
        self.rec_model = self.app.models['recognition']
        
    def get_embedding(self, img):
        if img is None: return None
        # img must be 112x112 aligned face
        feat = self.rec_model.get_feat(img)[0]
        # normalize
        return feat / np.linalg.norm(feat)

def evaluate_model_a_val(pairs_csv):
    engine = FastIdentityEngine()
    similarities = []
    labels = []
    
    with open(pairs_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in tqdm(list(reader), desc="Calibrating Model A"):
            ref_emb = np.load(row['reference_embedding_path'])
            probe_path = row['probe_path']
            
            img = cv2.imread(probe_path)
            probe_emb = engine.get_embedding(img)
            
            sim = 0.0
            if probe_emb is not None:
                sim = np.dot(ref_emb, probe_emb)
                
            similarities.append(sim)
            labels.append(1 if row['label'] == 'genuine' else 0)
            
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    tnr = 1 - fpr
    balanced_acc = (tpr + tnr) / 2
    best_idx = np.argmax(balanced_acc)
    best_threshold = float(thresholds[best_idx])
    
    with open('experiments/final_evaluation/model_a_threshold.json', 'w') as f:
        json.dump({
            "selected_threshold": best_threshold,
            "calibration_method": "max_balanced_accuracy",
            "validation_sample_count": len(similarities),
            "validation_genuine_count": sum(labels),
            "validation_impostor_count": len(labels) - sum(labels)
        }, f, indent=4)
        
    print(f"Model A calibrated threshold: {best_threshold:.4f}")
    return best_threshold

def evaluate_model_a_test(pairs_csv, threshold):
    engine = FastIdentityEngine()
    
    genuine_accepts = 0
    genuine_rejects = 0
    diff_person_accepts = 0
    imp_accepts = 0
    imp_rejects = 0
    
    total_gen = 0
    total_diff = 0
    total_imp = 0
    
    results = []
    
    with open(pairs_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in tqdm(list(reader), desc="Testing Model A"):
            ref_emb = np.load(row['reference_embedding_path'])
            probe_path = row['probe_path']
            
            img = cv2.imread(probe_path)
            probe_emb = engine.get_embedding(img)
            
            sim = 0.0
            if probe_emb is not None:
                sim = np.dot(ref_emb, probe_emb)
                
            match = sim >= threshold
            label = row['label']
            
            if label == 'genuine':
                total_gen += 1
                if match: genuine_accepts += 1
                else: genuine_rejects += 1
            elif label == 'different_person':
                total_diff += 1
                if match: diff_person_accepts += 1
            elif label == 'impersonation':
                total_imp += 1
                if match: imp_accepts += 1
                else: imp_rejects += 1
                
            results.append({
                'sample_id': row['pair_id'],
                'label': label,
                'sim': float(sim),
                'match': bool(match),
                'reference_identity': row['reference_identity'],
                'physical_identity': row['physical_identity']
            })
            
    tar = genuine_accepts / total_gen if total_gen else 0
    frr = genuine_rejects / total_gen if total_gen else 0
    far = (diff_person_accepts + imp_accepts) / (total_diff + total_imp) if (total_diff + total_imp) else 0
    
    imp_accept_rate = imp_accepts / total_imp if total_imp else 0
    diff_accept_rate = diff_person_accepts / total_diff if total_diff else 0
    
    metrics = {
        "accuracy": (genuine_accepts + (total_diff - diff_person_accepts) + (total_imp - imp_accepts)) / (total_gen + total_diff + total_imp),
        "TAR": tar,
        "FRR": frr,
        "FAR": far,
        "IMPERSONATION_ACCEPTANCE_RATE": imp_accept_rate,
        "impersonation_rejection_rate": imp_rejects / total_imp if total_imp else 0,
        "different_person_false_acceptance_rate": diff_accept_rate,
        "support": {
            "genuine": total_gen,
            "different_person": total_diff,
            "impersonation": total_imp
        }
    }
    
    with open('experiments/final_evaluation/model_a_test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    return results

def get_model_c_metrics(model, val_loader, device, criterion, epsilon):
    model.eval()
    all_preds = []
    all_labels = []
    
    for imgs, embs, labels in val_loader:
        imgs, embs, labels = imgs.to(device), embs.to(device), labels.to(device)
        
        if epsilon > 0:
            perturbed_imgs = fgsm_attack(model, imgs, embs, labels, criterion, epsilon)
            imgs = perturbed_imgs
            
        with torch.no_grad():
            logits = model(imgs, embs)
            preds = torch.argmax(logits, dim=1)
            
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        
    acc = float(accuracy_score(all_labels, all_preds))
    p, r, f, s = precision_recall_fscore_support(all_labels, all_preds, labels=[0, 1, 2], zero_division=0)
    mac_p, mac_r, mac_f, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    wgt_p, wgt_r, wgt_f, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2]).tolist()
    
    return {
        "accuracy": acc,
        "macro_precision": float(mac_p),
        "macro_recall": float(mac_r),
        "macro_f1": float(mac_f),
        "weighted_f1": float(wgt_f),
        "per_class": {
            "genuine": {"precision": float(p[0]), "recall": float(r[0]), "f1": float(f[0]), "support": int(s[0])},
            "different_person": {"precision": float(p[1]), "recall": float(r[1]), "f1": float(f[1]), "support": int(s[1])},
            "impersonation": {"precision": float(p[2]), "recall": float(r[2]), "f1": float(f[2]), "support": int(s[2])}
        },
        "confusion_matrix": cm,
        "predictions": [int(x) for x in all_preds]
    }

def main():
    val_csv = 'data/manifests/val_pairs.csv'
    test_csv = 'data/manifests/test_pairs.csv'
    
    print("1. Calibrating Model A on Validation Set...")
    threshold = evaluate_model_a_val(val_csv)
    
    print("\n2. Evaluating Model A on Test Set...")
    model_a_results = evaluate_model_a_test(test_csv, threshold)
    
    print("\n3. Evaluating Model C on Test Set...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    test_ds = AntiImpersonationDataset(test_csv)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    c_control = AntiImpersonationModel().to(device)
    c_control.load_state_dict(torch.load('experiments/model_c_control/best_model.pt', map_location=device, weights_only=True))
    c_control.eval()
    
    c_adv = AntiImpersonationModel().to(device)
    c_adv.load_state_dict(torch.load('experiments/model_c_adv/best_model.pt', map_location=device, weights_only=True))
    c_adv.eval()
    
    # 3a. Clean Evaluation
    print("Evaluating C-Control (clean)...")
    ctrl_clean = get_model_c_metrics(c_control, test_loader, device, criterion, 0.0)
    print("Evaluating C-Adv (clean)...")
    adv_clean = get_model_c_metrics(c_adv, test_loader, device, criterion, 0.0)
    
    with open('experiments/final_evaluation/model_c_test_metrics.json', 'w') as f:
        json.dump({"C-Control": ctrl_clean, "C-Adv": adv_clean}, f, indent=4)
        
    # 3b. Adversarial Evaluation
    epsilons = [0.01, 0.05, 0.10]
    adv_metrics = []
    
    for eps in epsilons:
        print(f"Evaluating C-Control (eps={eps})...")
        ctrl_adv = get_model_c_metrics(c_control, test_loader, device, criterion, eps)
        adv_metrics.append({"model": "C-Control", "epsilon": eps, "metrics": ctrl_adv})
        
        print(f"Evaluating C-Adv (eps={eps})...")
        adv_adv = get_model_c_metrics(c_adv, test_loader, device, criterion, eps)
        adv_metrics.append({"model": "C-Adv", "epsilon": eps, "metrics": adv_adv})
        
    with open('experiments/final_evaluation/adversarial_test_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'epsilon', 'accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1',
                         'genuine_precision', 'genuine_recall', 'genuine_f1',
                         'different_precision', 'different_recall', 'different_f1',
                         'impersonation_precision', 'impersonation_recall', 'impersonation_f1', 'support'])
        
        for record in adv_metrics:
            m = record['metrics']
            writer.writerow([
                record['model'], record['epsilon'], m['accuracy'], m['macro_precision'], m['macro_recall'], m['macro_f1'], m['weighted_f1'],
                m['per_class']['genuine']['precision'], m['per_class']['genuine']['recall'], m['per_class']['genuine']['f1'],
                m['per_class']['different_person']['precision'], m['per_class']['different_person']['recall'], m['per_class']['different_person']['f1'],
                m['per_class']['impersonation']['precision'], m['per_class']['impersonation']['recall'], m['per_class']['impersonation']['f1'],
                sum([m['per_class'][k]['support'] for k in m['per_class']])
            ])
            
    # Write Impersonation Robustness CSV
    with open('experiments/final_evaluation/impersonation_robustness.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'epsilon', 'impersonation_precision', 'impersonation_recall', 'impersonation_f1', 'impersonation_support'])
        
        writer.writerow(['C-Control', 0.0, ctrl_clean['per_class']['impersonation']['precision'], ctrl_clean['per_class']['impersonation']['recall'], ctrl_clean['per_class']['impersonation']['f1'], ctrl_clean['per_class']['impersonation']['support']])
        for record in adv_metrics:
            if record['model'] == 'C-Control':
                writer.writerow(['C-Control', record['epsilon'], record['metrics']['per_class']['impersonation']['precision'], record['metrics']['per_class']['impersonation']['recall'], record['metrics']['per_class']['impersonation']['f1'], record['metrics']['per_class']['impersonation']['support']])
        
        writer.writerow(['C-Adv', 0.0, adv_clean['per_class']['impersonation']['precision'], adv_clean['per_class']['impersonation']['recall'], adv_clean['per_class']['impersonation']['f1'], adv_clean['per_class']['impersonation']['support']])
        for record in adv_metrics:
            if record['model'] == 'C-Adv':
                writer.writerow(['C-Adv', record['epsilon'], record['metrics']['per_class']['impersonation']['precision'], record['metrics']['per_class']['impersonation']['recall'], record['metrics']['per_class']['impersonation']['f1'], record['metrics']['per_class']['impersonation']['support']])

    print("\n4. Running Offline Final Decision Engine & Error Analysis...")
    # Decision logic:
    # 1. Model A verification (match/reject)
    # 2. Model C anti-impersonation classification (0: genuine, 1: diff, 2: imp)
    
    # Let's align model_a_results and ctrl_clean predictions (they iterate in the same order over test_pairs.csv)
    c_preds = ctrl_clean['predictions']
    
    with open('experiments/final_evaluation/final_decision_results.csv', 'w', newline='') as f1, \
         open('experiments/final_evaluation/error_analysis.csv', 'w', newline='') as f2:
        
        dec_writer = csv.writer(f1)
        dec_writer.writerow(['sample_id', 'reference_identity', 'physical_identity', 'model_a_match', 'model_a_similarity', 'model_c_prediction', 'final_state', 'ground_truth_category', 'correct_final_state'])
        
        err_writer = csv.writer(f2)
        err_writer.writerow(['sample_id', 'ground_truth', 'model_a_result', 'model_c_result', 'final_state', 'reference_identity', 'physical_identity', 'attack_type', 'notes'])
        
        states_true = []
        states_pred = []
        
        for i, a_res in enumerate(model_a_results):
            c_pred = c_preds[i]
            
            # Ground truth state
            gt_cat = a_res['label']
            if gt_cat == 'genuine': true_state = 'VERIFIED'
            elif gt_cat == 'impersonation': true_state = 'SUSPECTED_IMPERSONATION'
            else: true_state = 'UNKNOWN'
            
            # Predicted state
            if not a_res['match']:
                final_state = 'UNKNOWN'
            else:
                if c_pred == 0: final_state = 'VERIFIED'
                elif c_pred == 2: final_state = 'SUSPECTED_IMPERSONATION'
                else: final_state = 'UNKNOWN'  # c_pred == 1
                
            correct = (final_state == true_state)
            states_true.append(true_state)
            states_pred.append(final_state)
            
            dec_writer.writerow([
                a_res['sample_id'], a_res['reference_identity'], a_res['physical_identity'],
                a_res['match'], a_res['sim'], c_pred, final_state, gt_cat, correct
            ])
            
            if not correct:
                # categorize error
                note = "Other"
                if gt_cat == 'genuine' and final_state == 'UNKNOWN':
                    if not a_res['match']: note = "Identity false rejection"
                    elif c_pred == 1: note = "Genuine confused with different_person"
                elif gt_cat == 'different_person' and final_state == 'VERIFIED':
                    note = "Identity false acceptance & confused with genuine"
                elif gt_cat == 'impersonation' and final_state == 'VERIFIED':
                    note = "Impersonation missed (Accepted as Genuine)"
                elif gt_cat == 'impersonation' and final_state == 'UNKNOWN':
                    if not a_res['match']: note = "Impersonation missed by Model A (Failed identity match)"
                    elif c_pred == 1: note = "Impersonation confused with different_person"
                    
                err_writer.writerow([
                    a_res['sample_id'], gt_cat, 'MATCH' if a_res['match'] else 'REJECT', 
                    c_pred, final_state, a_res['reference_identity'], a_res['physical_identity'], 
                    'FaceSwap', note
                ])
                
    # Final Decision Metrics
    with open('experiments/final_evaluation/security_comparison.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'genuine_acceptance_rate', 'different_person_far', 'impersonation_acceptance_rate'])
        
        # We'll just put the Model A stats and the Final System stats
        with open('experiments/final_evaluation/model_a_test_metrics.json') as mf:
            m_a = json.load(mf)
        
        writer.writerow(['Model A Only', m_a['TAR'], m_a['different_person_false_acceptance_rate'], m_a['IMPERSONATION_ACCEPTANCE_RATE']])
        
        sys_tar = sum(1 for t, p in zip(states_true, states_pred) if t == 'VERIFIED' and p == 'VERIFIED') / sum(1 for t in states_true if t == 'VERIFIED')
        sys_diff_far = sum(1 for a_res, p in zip(model_a_results, states_pred) if a_res['label'] == 'different_person' and p == 'VERIFIED') / sum(1 for a_res in model_a_results if a_res['label'] == 'different_person')
        sys_imp_far = sum(1 for a_res, p in zip(model_a_results, states_pred) if a_res['label'] == 'impersonation' and p == 'VERIFIED') / sum(1 for a_res in model_a_results if a_res['label'] == 'impersonation')
        
        writer.writerow(['Model A + C-Control', sys_tar, sys_diff_far, sys_imp_far])

    print("\n5. Computing Bootstrapped Confidence Intervals...")
    def bootstrap_metric(y_true, y_pred, metric_fn, n_bootstraps=1000):
        rng = np.random.RandomState(42)
        metrics = []
        n = len(y_true)
        for _ in range(n_bootstraps):
            indices = rng.randint(0, n, n)
            metrics.append(metric_fn(np.array(y_true)[indices], np.array(y_pred)[indices]))
        return np.percentile(metrics, [2.5, 97.5])
    
    def imp_f1_metric(yt, yp):
        p, r, f, _ = precision_recall_fscore_support(yt, yp, labels=[0, 1, 2], zero_division=0)
        return f[2]
        
    def imp_acc_rate_metric(yt, yp):
        # yt is labels (impersonation = 2), yp is match (bool)
        imp_idx = yt == 2
        if not np.any(imp_idx): return 0
        return np.sum(yp[imp_idx]) / np.sum(imp_idx)
        
    with open('experiments/final_evaluation/confidence_intervals.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value', 'ci_lower_95', 'ci_upper_95'])
        
        # Model A impersonation acceptance rate
        yt_a = np.array([2 if r['label'] == 'impersonation' else (1 if r['label'] == 'different_person' else 0) for r in model_a_results])
        yp_a = np.array([1 if r['match'] else 0 for r in model_a_results])
        val = imp_acc_rate_metric(yt_a, yp_a)
        ci = bootstrap_metric(yt_a, yp_a, imp_acc_rate_metric)
        writer.writerow(['Model A Impersonation Acceptance Rate', val, ci[0], ci[1]])
        
        # C-Control eps=0
        yt_c = np.array(ctrl_clean['predictions']) # wait, yt_c should be true labels
        yt_true = [2 if r['label'] == 'impersonation' else (1 if r['label'] == 'different_person' else 0) for r in model_a_results]
        
        val = imp_f1_metric(yt_true, ctrl_clean['predictions'])
        ci = bootstrap_metric(yt_true, ctrl_clean['predictions'], imp_f1_metric)
        writer.writerow(['C-Control Impersonation F1 (eps=0.0)', val, ci[0], ci[1]])
        
        # C-Control eps=0.05
        preds = [record['metrics']['predictions'] for record in adv_metrics if record['model'] == 'C-Control' and record['epsilon'] == 0.05][0]
        val = imp_f1_metric(yt_true, preds)
        ci = bootstrap_metric(yt_true, preds, imp_f1_metric)
        writer.writerow(['C-Control Impersonation F1 (eps=0.05)', val, ci[0], ci[1]])
        
        # C-Control eps=0.10
        preds = [record['metrics']['predictions'] for record in adv_metrics if record['model'] == 'C-Control' and record['epsilon'] == 0.10][0]
        val = imp_f1_metric(yt_true, preds)
        ci = bootstrap_metric(yt_true, preds, imp_f1_metric)
        writer.writerow(['C-Control Impersonation F1 (eps=0.10)', val, ci[0], ci[1]])
        
        # C-Adv eps=0
        val = imp_f1_metric(yt_true, adv_clean['predictions'])
        ci = bootstrap_metric(yt_true, adv_clean['predictions'], imp_f1_metric)
        writer.writerow(['C-Adv Impersonation F1 (eps=0.0)', val, ci[0], ci[1]])
        
        # C-Adv eps=0.05
        preds = [record['metrics']['predictions'] for record in adv_metrics if record['model'] == 'C-Adv' and record['epsilon'] == 0.05][0]
        val = imp_f1_metric(yt_true, preds)
        ci = bootstrap_metric(yt_true, preds, imp_f1_metric)
        writer.writerow(['C-Adv Impersonation F1 (eps=0.05)', val, ci[0], ci[1]])
        
        # C-Adv eps=0.10
        preds = [record['metrics']['predictions'] for record in adv_metrics if record['model'] == 'C-Adv' and record['epsilon'] == 0.10][0]
        val = imp_f1_metric(yt_true, preds)
        ci = bootstrap_metric(yt_true, preds, imp_f1_metric)
        writer.writerow(['C-Adv Impersonation F1 (eps=0.10)', val, ci[0], ci[1]])

    print("\nPhase 5 Evaluation Complete.")

if __name__ == "__main__":
    main()
