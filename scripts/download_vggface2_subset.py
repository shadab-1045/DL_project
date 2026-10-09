import os
import json
import random
from datasets import load_dataset

def download_vggface2_subset():
    target_identities = 50
    images_per_identity = 40
    output_dir = "data/raw/vggface2"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Streaming from HuggingFace logasja/VGGFace2...")
    ds = load_dataset('logasja/VGGFace2', split='train', streaming=True)
    features = ds.features['class_id']
    
    counts = {}
    completed = set()
    
    print(f"Collecting up to {images_per_identity} images for {target_identities} identities...")
    
    for item in ds:
        class_idx = item['class_id']
        name = features.int2str(class_idx)
        
        if name in completed:
            continue
            
        if name not in counts:
            if len(counts) >= target_identities:
                # We have already started collecting target_identities, skip new ones
                continue
            counts[name] = 0
            
        if counts[name] < images_per_identity:
            id_dir = os.path.join(output_dir, name)
            os.makedirs(id_dir, exist_ok=True)
            
            img = item['image']
            out_path = os.path.join(id_dir, f"{counts[name]:04d}.jpg")
            img.save(out_path)
            
            counts[name] += 1
            
            if counts[name] == images_per_identity:
                completed.add(name)
                print(f"Completed identity {name} ({len(completed)}/{target_identities})")
                
                if len(completed) == target_identities:
                    print("Finished collecting all required images.")
                    break
                    
    total_images = sum(counts.values())
    print(f"Downloaded {total_images} images across {len(counts)} identities.")
    
    print("Building identity split...")
    identities = sorted(list(counts.keys()))
    
    random.seed(42)
    random.shuffle(identities)
    
    train_ids = identities[:30]
    val_ids = identities[30:40]
    test_ids = identities[40:50]
    
    split = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }
    
    split_path = "data/manifests/identity_split.json"
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    with open(split_path, 'w') as f:
        json.dump(split, f, indent=4)
        
    print(f"Saved identity split to {split_path}")
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

if __name__ == "__main__":
    download_vggface2_subset()
