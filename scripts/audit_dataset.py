import os
import json
import csv

def audit_dataset():
    print("Starting dataset leakage and integrity audit...")
    split_file = "data/manifests/identity_split.json"
    if not os.path.exists(split_file):
        print("Identity split file missing.")
        return False
        
    with open(split_file, "r") as f:
        splits = json.load(f)
        
    train_ids = set(splits.get("train_identities", []))
    val_ids = set(splits.get("validation_identities", []))
    test_ids = set(splits.get("test_identities", []))
    
    # 1. Check identity leakage
    if len(train_ids & val_ids) > 0:
        print("FAIL: Identity leakage between train and val")
        return False
    if len(train_ids & test_ids) > 0:
        print("FAIL: Identity leakage between train and test")
        return False
    if len(val_ids & test_ids) > 0:
        print("FAIL: Identity leakage between val and test")
        return False
        
    print("PASS: No identity leakage between splits.")
    
    # 2. Check metadata vs logical assignments
    manifests = {
        "train": "data/manifests/train_pairs.csv",
        "validation": "data/manifests/val_pairs.csv",
        "test": "data/manifests/test_pairs.csv"
    }
    
    for split_name, manifest_path in manifests.items():
        if not os.path.exists(manifest_path):
            continue
            
        expected_ids = train_ids if split_name == "train" else (val_ids if split_name == "validation" else test_ids)
        
        with open(manifest_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Check reference identity
                if row["reference_identity"] not in expected_ids:
                    print(f"FAIL: Reference identity {row['reference_identity']} not in {split_name} split.")
                    return False
                    
                # Check physical identity
                if row["physical_identity"] not in expected_ids:
                    print(f"FAIL: Physical identity {row['physical_identity']} not in {split_name} split.")
                    return False
                    
                # Check label logic
                if row["label"] == "genuine" and row["reference_identity"] != row["physical_identity"]:
                    print("FAIL: Genuine pair has different reference and physical identities.")
                    return False
                if row["label"] == "different_person" and row["reference_identity"] == row["physical_identity"]:
                    print("FAIL: Different person pair has same reference and physical identities.")
                    return False
                if row["label"] == "impersonation" and row["reference_identity"] == row["physical_identity"]:
                    print("FAIL: Impersonation pair uses same person for source and target.")
                    return False
                    
    print("PASS: Logical assignments are correct.")
    return True

if __name__ == "__main__":
    success = audit_dataset()
    if not success:
        exit(1)
