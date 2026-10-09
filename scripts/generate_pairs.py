import os
import json
import csv
import random

def generate_pairs_for_split(split_name, identities, raw_data_dir, out_file, images_per_id=25):
    pairs = []
    pair_id_counter = 1
    
    # Collect images for each identity
    id_to_images = {}
    for identity in identities:
        id_dir = os.path.join(raw_data_dir, identity)
        images = [f for f in os.listdir(id_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(images) >= images_per_id:
            id_to_images[identity] = random.sample(images, images_per_id)
        else:
            id_to_images[identity] = images
            
    # Generate genuine pairs
    for identity, images in id_to_images.items():
        if len(images) < 2: continue
        
        # Pick one reference, others as probe
        ref_img = images[0]
        for probe_img in images[1:3]: # Create a couple of genuine pairs
            pairs.append({
                "pair_id": f"{split_name}_{pair_id_counter}",
                "reference_identity": identity,
                "reference_embedding_path": os.path.join("data/processed/faces", identity, ref_img),
                "probe_sample_id": f"s_{pair_id_counter}",
                "probe_path": os.path.join(raw_data_dir, identity, probe_img),
                "visible_identity": identity,
                "physical_identity": identity,
                "label": "genuine",
                "attack_type": "none",
                "source_dataset": "vggface2",
                "generator": "none",
                "split": split_name
            })
            pair_id_counter += 1
            
    # Generate different_person pairs
    for identity, images in id_to_images.items():
        if not images: continue
        ref_img = images[0]
        
        # Pick a different identity
        other_id = random.choice([i for i in identities if i != identity])
        if not id_to_images[other_id]: continue
        
        probe_img = random.choice(id_to_images[other_id])
        
        pairs.append({
            "pair_id": f"{split_name}_{pair_id_counter}",
            "reference_identity": identity,
            "reference_embedding_path": os.path.join("data/processed/faces", identity, ref_img),
            "probe_sample_id": f"s_{pair_id_counter}",
            "probe_path": os.path.join(raw_data_dir, other_id, probe_img),
            "visible_identity": other_id,
            "physical_identity": other_id,
            "label": "different_person",
            "attack_type": "none",
            "source_dataset": "vggface2",
            "generator": "none",
            "split": split_name
        })
        pair_id_counter += 1
        
    # Generate impersonation pairs (face swap configs)
    for identity, images in id_to_images.items():
        if not images: continue
        ref_img = images[0] # Source face for the swap
        
        # Pick a physical target identity
        other_id = random.choice([i for i in identities if i != identity])
        if not id_to_images[other_id]: continue
        
        target_img = random.choice(id_to_images[other_id])
        
        pairs.append({
            "pair_id": f"{split_name}_{pair_id_counter}",
            "reference_identity": identity,
            "reference_embedding_path": os.path.join("data/processed/faces", identity, ref_img),
            "probe_sample_id": f"s_{pair_id_counter}",
            "probe_path": os.path.join("data/generated/face_swaps", f"swap_{pair_id_counter}.jpg"),
            "visible_identity": identity,
            "physical_identity": other_id,
            "label": "impersonation",
            "attack_type": "face_swap",
            "source_dataset": "vggface2",
            "generator": "inswapper",
            "split": split_name,
            # We also need to store source/target for the swapper script
            "_swap_src": os.path.join(raw_data_dir, identity, ref_img),
            "_swap_dst": os.path.join(raw_data_dir, other_id, target_img)
        })
        pair_id_counter += 1
        
    # Save to CSV
    keys = ["pair_id", "reference_identity", "reference_embedding_path", "probe_sample_id", 
            "probe_path", "visible_identity", "physical_identity", "label", "attack_type",
            "source_dataset", "generator", "split", "_swap_src", "_swap_dst"]
    
    with open(out_file, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(pairs)
        
    print(f"Generated {len(pairs)} pairs for {split_name} split.")

def main():
    split_file = "data/manifests/identity_split.json"
    if not os.path.exists(split_file):
        print("Identity split file not found.")
        return
        
    with open(split_file, "r") as f:
        splits = json.load(f)
        
    raw_dir = "data/raw/vggface2"
    generate_pairs_for_split("train", splits["train_identities"], raw_dir, "data/manifests/train_pairs.csv")
    generate_pairs_for_split("validation", splits["validation_identities"], raw_dir, "data/manifests/val_pairs.csv")
    generate_pairs_for_split("test", splits["test_identities"], raw_dir, "data/manifests/test_pairs.csv")

if __name__ == "__main__":
    main()
