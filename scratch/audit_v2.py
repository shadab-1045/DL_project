import os
import csv
import cv2
import hashlib
import numpy as np
import insightface
from collections import defaultdict
from tqdm import tqdm

def sha256(filepath):
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def check_v1_immutability():
    print("--- V1 IMMUTABILITY ---")
    files_to_check = [
        "data/manifests/train_pairs.csv",
        "data/manifests/val_pairs.csv",
        "data/manifests/test_pairs.csv",
    ]
    for f in files_to_check:
        if os.path.exists(f):
            print(f"{f}: modified {os.stat(f).st_mtime}")
        else:
            print(f"{f}: MISSING")
            
def run_audit():
    manifests = {
        "train": "data/manifests/train_pairs_v2.csv",
        "val": "data/manifests/val_pairs_v2.csv",
        "test": "data/manifests/test_pairs_v2.csv"
    }
    
    data = {}
    for split, path in manifests.items():
        with open(path, "r") as f:
            data[split] = list(csv.DictReader(f))
            
    # 1. SPLIT INTEGRITY
    print("\n--- 1. SPLIT INTEGRITY ---")
    ref_ids = {}
    phys_ids = {}
    for split, rows in data.items():
        ref_ids[split] = set(r["reference_identity"] for r in rows)
        phys_ids[split] = set(r["physical_identity"] for r in rows)
        print(f"{split} Ref IDs: {len(ref_ids[split])}, Phys IDs: {len(phys_ids[split])}")
        
    print(f"train intersect val ref: {ref_ids['train'].intersection(ref_ids['val'])}")
    print(f"train intersect test ref: {ref_ids['train'].intersection(ref_ids['test'])}")
    print(f"val intersect test ref: {ref_ids['val'].intersection(ref_ids['test'])}")
    
    print(f"train intersect val phys: {phys_ids['train'].intersection(phys_ids['val'])}")
    print(f"train intersect test phys: {phys_ids['train'].intersection(phys_ids['test'])}")
    print(f"val intersect test phys: {phys_ids['val'].intersection(phys_ids['test'])}")
    
    # 2. DUPLICATION & 3. CLASS / IMAGE INTEGRITY & 4. ALIGNMENT PARITY
    print("\n--- 2, 3, 4. DUPLICATION & INTEGRITY & ALIGNMENT ---")
    app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    for split, rows in data.items():
        print(f"\nEvaluating {split}...")
        unique_paths = set()
        hashes = set()
        duplicate_paths = 0
        duplicate_hashes = 0
        
        counts = {"genuine": 0, "different_person": 0, "impersonation": 0}
        issues = 0
        
        # for alignment parity
        kps_means = {"genuine": [], "different_person": [], "impersonation": []}
        
        for r in tqdm(rows, desc=split):
            path = r["probe_path"]
            label = r["label"]
            counts[label] += 1
            
            if path in unique_paths:
                duplicate_paths += 1
            unique_paths.add(path)
            
            hsh = sha256(path)
            if hsh in hashes:
                duplicate_hashes += 1
            hashes.add(hsh)
            
            img = cv2.imread(path)
            if img is None:
                issues += 1
                continue
                
            if img.shape != (112, 112, 3):
                issues += 1
                
            if np.mean(img) < 1.0:
                issues += 1
                
            # Alignment parity subset check (take subset to speed up)
            if counts[label] < 20: 
                # To help detection of tightly cropped face, we could pad it, or detect on raw
                faces = app.get(img)
                if faces:
                    best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                    kps_means[label].append(best.kps)
                    
        print(f"Total Rows: {len(rows)}, Unique Paths: {len(unique_paths)}, Dup Paths: {duplicate_paths}, Dup Hashes: {duplicate_hashes}")
        print(f"Counts: {counts}, Issues: {issues}")
        
        for lbl in kps_means:
            if kps_means[lbl]:
                mean_kps = np.mean(kps_means[lbl], axis=0)
                print(f"Alignment Parity {lbl} (n={len(kps_means[lbl])}): LeftEye {mean_kps[0]}, RightEye {mean_kps[1]}")
            else:
                print(f"Alignment Parity {lbl}: No faces detected natively (tight crop limits detection)")

    # 6. SOURCE/TARGET PAIRING
    print("\n--- 6. SOURCE/TARGET PAIRING ---")
    for split, rows in data.items():
        impersonation_rows = [r for r in rows if r["label"] == "impersonation"]
        self_swap = sum(1 for r in impersonation_rows if r["reference_identity"] == r["physical_identity"])
        print(f"[{split}] Impersonation samples: {len(impersonation_rows)}")
        print(f"[{split}] Self-swaps (ref == phys): {self_swap}")
        targets = [r["physical_identity"] for r in impersonation_rows]
        if targets:
            unique, counts = np.unique(targets, return_counts=True)
            print(f"[{split}] Unique targets: {len(unique)}, Max reuse of single target: {counts.max()}")

    # 8. MANIFEST SEMANTICS
    print("\n--- 8. MANIFEST SEMANTICS ---")
    print(f"Columns: {list(data['train'][0].keys())}")
    
    for lbl in ["genuine", "different_person", "impersonation"]:
        for r in data['train']:
            if r["label"] == lbl:
                print(f"Sample {lbl}: Ref {r['reference_identity']} vs Phys {r['physical_identity']} | probe {r['probe_path']}")
                break

if __name__ == "__main__":
    check_v1_immutability()
    run_audit()
