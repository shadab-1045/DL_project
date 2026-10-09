import csv
import os

def check_integrity():
    with open('experiments/data_qa/human_visual_direction_qa.csv', 'r') as f:
        qa_data = list(csv.DictReader(f))
        
    with open('data/manifests/samples.csv', 'r') as f:
        samples_data = list(csv.DictReader(f))
        
    samples_dict = {s['sample_id']: s for s in samples_data}
    
    seen_ids = set()
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    
    # Pre-calculate genuine by id for source image check
    genuine_by_id = {}
    for s in samples_data:
        if s['attack_type'] == 'none':
            if s['physical_identity'] not in genuine_by_id:
                genuine_by_id[s['physical_identity']] = s['context_path']

    for row in qa_data:
        sid = row['sample_id']
        
        # duplicates check
        if sid in seen_ids:
            raise ValueError(f"Duplicate QA sample ID: {sid}")
        seen_ids.add(sid)
        
        split_counts[row['split']] += 1
        
        # sample_id in samples.csv
        if sid not in samples_dict:
            raise ValueError(f"QA sample_id {sid} not found in samples.csv")
            
        s = samples_dict[sid]
        
        # metadata match
        if s['visible_identity'] != row['source_identity']:
            raise ValueError(f"Source identity mismatch for {sid}")
        if s['physical_identity'] != row['target_identity']:
            raise ValueError(f"Target identity mismatch for {sid}")
        if s['split'] != row['split']:
            raise ValueError(f"Split mismatch for {sid}")
        if s['attack_type'] != 'face_swap':
            raise ValueError(f"Attack type mismatch for {sid}")
        if s['generator'] != 'inswapper_128':
            raise ValueError(f"Generator mismatch for {sid}")
            
        # image missing checks
        if not os.path.exists(s['path']):
            raise ValueError(f"Generated image missing for {sid}: {s['path']}")
        
        if not os.path.exists(s['context_path']):
            raise ValueError(f"Target image missing for {sid}: {s['context_path']}")
            
        src_img = genuine_by_id.get(s['visible_identity'])
        if not src_img or not os.path.exists(src_img):
            raise ValueError(f"Source image missing for identity {s['visible_identity']}")
            
    # split counts check
    if split_counts['train'] != 4 or split_counts['val'] != 3 or split_counts['test'] != 3:
        raise ValueError(f"Split counts are wrong! Found: {split_counts}")
        
    print("ALL INTEGRITY CHECKS PASSED.")

if __name__ == "__main__":
    check_integrity()
