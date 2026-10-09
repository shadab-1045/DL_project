import os
import json
import random
from collections import defaultdict
import glob

def build_identity_split(raw_data_dir, out_file, seed=42):
    random.seed(seed)
    
    # Raw data directory should have one folder per identity
    if not os.path.exists(raw_data_dir):
        print(f"Data directory {raw_data_dir} does not exist.")
        return False
        
    identities = [d for d in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, d))]
    
    if len(identities) < 360:
        print(f"Not enough identities found (found {len(identities)}, need at least 360).")
        return False
        
    random.shuffle(identities)
    
    train_ids = identities[:300]
    val_ids = identities[300:330]
    test_ids = identities[330:360]
    
    # Audit for leakage
    assert len(set(train_ids) & set(val_ids)) == 0
    assert len(set(train_ids) & set(test_ids)) == 0
    assert len(set(val_ids) & set(test_ids)) == 0
    
    split_info = {
        "seed": seed,
        "dataset_version": "vggface2",
        "counts": {
            "train": len(train_ids),
            "validation": len(val_ids),
            "test": len(test_ids)
        },
        "train_identities": train_ids,
        "validation_identities": val_ids,
        "test_identities": test_ids
    }
    
    with open(out_file, "w") as f:
        json.dump(split_info, f, indent=4)
        
    print(f"Created identity split successfully. Saved to {out_file}")
    return True

if __name__ == "__main__":
    build_identity_split("data/raw/vggface2", "data/manifests/identity_split.json")
