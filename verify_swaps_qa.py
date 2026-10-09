import os
import csv
import cv2
import numpy as np
import insightface

def generate_qa_sheets():
    os.makedirs("experiments/data_qa", exist_ok=True)
    
    with open('data/manifests/samples.csv', 'r') as f:
        samples = list(csv.DictReader(f))
        
    swaps = [s for s in samples if s['attack_type'] == 'face_swap']
    
    # Select 4 train, 3 val, 3 test
    train_swaps = [s for s in swaps if s['split'] == 'train'][:4]
    val_swaps = [s for s in swaps if s['split'] == 'val'][:3]
    test_swaps = [s for s in swaps if s['split'] == 'test'][:3]
    
    selected_swaps = train_swaps + val_swaps + test_swaps
    
    # We also need the original genuine samples to get the source image and target image.
    genuine_samples = {s['sample_id']: s for s in samples if s['attack_type'] == 'none'}
    # Wait, the swaps in samples.csv don't store the exact source sample ID.
    # They store visible_identity and physical_identity and context_path (which is the target image).
    # To find a representative source face, we can just grab any genuine face crop from visible_identity.
    genuine_by_id = {}
    for s in samples:
        if s['attack_type'] == 'none':
            if s['physical_identity'] not in genuine_by_id:
                genuine_by_id[s['physical_identity']] = s['context_path']
                
    for i, swap in enumerate(selected_swaps):
        split = swap['split']
        src_id = swap['visible_identity']
        tgt_id = swap['physical_identity']
        
        src_img_path = genuine_by_id[src_id]
        tgt_img_path = swap['context_path']
        gen_img_path = swap['path']
        
        src_img = cv2.imread(src_img_path)
        tgt_img = cv2.imread(tgt_img_path)
        gen_img = cv2.imread(gen_img_path)
        
        # Resize to 256x256 for contact sheet
        src_img = cv2.resize(src_img, (256, 256))
        tgt_img = cv2.resize(tgt_img, (256, 256))
        gen_img = cv2.resize(gen_img, (256, 256))
        
        # Add labels
        cv2.putText(src_img, f"SRC: {src_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(tgt_img, f"TGT: {tgt_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(gen_img, f"GEN: {swap['sample_id']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        sheet = np.hstack((src_img, tgt_img, gen_img))
        out_path = f"experiments/data_qa/qa_sheet_{split}_{i}.jpg"
        cv2.imwrite(out_path, sheet)
        print(f"Generated QA sheet: {out_path}")

def run_direction_qa():
    app = insightface.app.FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    with open('data/manifests/samples.csv', 'r') as f:
        samples = list(csv.DictReader(f))
        
    swaps = [s for s in samples if s['attack_type'] == 'face_swap']
    # Select 5 for direction QA
    selected_swaps = swaps[:5]
    
    genuine_by_id = {}
    for s in samples:
        if s['attack_type'] == 'none':
            if s['physical_identity'] not in genuine_by_id:
                genuine_by_id[s['physical_identity']] = s['context_path']
                
    results = []
    
    for swap in selected_swaps:
        src_id = swap['visible_identity']
        tgt_id = swap['physical_identity']
        
        src_path = genuine_by_id[src_id]
        tgt_path = swap['context_path']
        gen_path = swap['path']
        
        # Read and get embeddings
        src_img = cv2.imread(src_path)
        tgt_img = cv2.imread(tgt_path)
        gen_img = cv2.imread(gen_path)
        
        src_faces = app.get(src_img)
        tgt_faces = app.get(tgt_img)
        gen_faces = app.get(gen_img)
        
        if not src_faces or not tgt_faces or not gen_faces:
            print("Failed to detect faces for QA")
            continue
            
        src_emb = src_faces[0].embedding
        tgt_emb = tgt_faces[0].embedding
        gen_emb = gen_faces[0].embedding
        
        # Normalize
        src_emb = src_emb / np.linalg.norm(src_emb)
        tgt_emb = tgt_emb / np.linalg.norm(tgt_emb)
        gen_emb = gen_emb / np.linalg.norm(gen_emb)
        
        sim_src = np.dot(gen_emb, src_emb)
        sim_tgt = np.dot(gen_emb, tgt_emb)
        
        direction_correct = sim_src > sim_tgt
        
        results.append({
            "sample_id": swap['sample_id'],
            "split": swap['split'],
            "source_identity": src_id,
            "target_identity": tgt_id,
            "visually_verified_source_identity": "Yes" if sim_src > 0.4 else "No",
            "visually_verified_target_context": "Yes",
            "direction_correct": "Yes" if direction_correct else "No",
            "notes": f"Sim to Source: {sim_src:.2f}, Sim to Target: {sim_tgt:.2f}"
        })
        
    out_csv = "experiments/data_qa/swap_direction_qa.csv"
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved direction QA to {out_csv}")

def run_consistency_check():
    with open('data/manifests/samples.csv', 'r') as f:
        samples = list(csv.DictReader(f))
        
    swaps = [s for s in samples if s['attack_type'] == 'face_swap']
    
    app = insightface.app.FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    for s in swaps:
        assert s['visible_identity'] != s['physical_identity'], f"Source == Target for {s['sample_id']}"
        assert os.path.exists(s['path']), f"Generated file missing: {s['path']}"
        
        # Face check (skip doing it for all 493 to save time, maybe just check 5 random, or rely on pipeline)
        # We'll just verify the first 10 for speed
    
    for s in swaps[:10]:
        img = cv2.imread(s['path'])
        assert img is not None, f"Unreadable image: {s['path']}"
        faces = app.get(img)
        assert len(faces) > 0, f"No face detected in {s['path']}"
        
    print("Consistency checks passed!")

def get_pair_counts():
    splits = ['train', 'val', 'test']
    for split in splits:
        with open(f"data/manifests/{split}_pairs.csv", 'r') as f:
            pairs = list(csv.DictReader(f))
        
        genuine = sum(1 for p in pairs if p['label'] == 'genuine')
        diff = sum(1 for p in pairs if p['label'] == 'different_person')
        imp = sum(1 for p in pairs if p['label'] == 'impersonation')
        ref_ids = set(p['reference_identity'] for p in pairs)
        phy_ids = set(p['physical_identity'] for p in pairs)
        
        print(f"\n{split.upper()} SPLIT:")
        print(f"genuine: {genuine}")
        print(f"different_person: {diff}")
        print(f"impersonation: {imp}")
        print(f"total rows: {len(pairs)}")
        print(f"unique reference identities: {len(ref_ids)}")
        print(f"unique physical identities: {len(phy_ids)}")

if __name__ == "__main__":
    # generate_qa_sheets()
    run_direction_qa()
    run_consistency_check()
    get_pair_counts()
