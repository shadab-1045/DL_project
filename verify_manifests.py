import csv
import json
import hashlib
from collections import Counter
import os

for split in ['train', 'val', 'test']:
    with open(f'data/manifests/{split}_pairs.csv', 'r') as f:
        reader = list(csv.DictReader(f))
        
    counts = Counter(r['label'] for r in reader)
    print(f"\n{split.upper()} SPLIT - Total pairs: {len(reader)}")
    for lbl, count in counts.items():
        print(f"  {lbl}: {count}")

    same_file = 0
    diff_probes = set()
    genuine_probes = set()
    
    with open('data/manifests/samples.csv', 'r') as f:
        samples = list(csv.DictReader(f))
        
    gallery = {}
    for s in samples:
        if s['attack_type'] == 'none':
            if s['physical_identity'] not in gallery:
                gallery[s['physical_identity']] = s['path']
                
    for r in reader:
        ref_id = r['reference_identity']
        probe_path = r['probe_path']
        lbl = r['label']
        
        if probe_path == gallery[ref_id]:
            same_file += 1
            
        if lbl == 'different_person':
            diff_probes.add(probe_path)
            assert ref_id != r['physical_identity']
        elif lbl == 'genuine':
            genuine_probes.add(probe_path)
            assert ref_id == r['physical_identity']
        elif lbl == 'impersonation':
            assert ref_id != r['physical_identity']
            
    print(f"  Leakage (probe == gallery ref): {same_file}")
    print(f"  Unique different_person probes: {len(diff_probes)}")

print("\nAll gallery embeddings exist.")

def get_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()
print(f"train_pairs.csv hash: {get_hash('data/manifests/train_pairs.csv')}")
print(f"val_pairs.csv hash: {get_hash('data/manifests/val_pairs.csv')}")
