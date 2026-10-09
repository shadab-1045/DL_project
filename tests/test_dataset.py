import os
import csv
import json
import cv2
import pytest

MANIFEST_DIR = "data/manifests/"

def get_samples():
    with open(os.path.join(MANIFEST_DIR, "samples.csv"), 'r') as f:
        return list(csv.DictReader(f))
        
def get_pairs():
    pairs = []
    for split in ["train", "val", "test"]:
        p = os.path.join(MANIFEST_DIR, f"{split}_pairs.csv")
        if os.path.exists(p):
            with open(p, 'r') as f:
                pairs.extend(list(csv.DictReader(f)))
    return pairs

def test_manifest_schema():
    samples = get_samples()
    assert len(samples) > 0
    expected_keys = {"sample_id", "path", "context_path", "visible_identity", "physical_identity", 
                     "attack_type", "source_dataset", "generator", "video_id", "frame_id", "split"}
    assert set(samples[0].keys()) == expected_keys

def test_path_validation():
    samples = get_samples()
    for s in samples:
        assert os.path.exists(s["path"]), f"Path missing: {s['path']}"
        assert os.path.exists(s["context_path"]), f"Context path missing: {s['context_path']}"

def test_label_validation():
    pairs = get_pairs()
    valid_labels = {"genuine", "different_person", "impersonation"}
    for p in pairs:
        assert p["label"] in valid_labels

def test_identity_split_disjointness():
    with open(os.path.join(MANIFEST_DIR, "identity_split.json"), 'r') as f:
        splits = json.load(f)
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

def test_no_mock_swaps():
    samples = get_samples()
    for s in samples:
        assert s["generator"] != "mock_simswap", f"Mock swap found in dataset: {s}"

def test_swap_metadata_correctness():
    samples = get_samples()
    swaps = [s for s in samples if s["attack_type"] == "face_swap"]
    for s in swaps:
        assert s["visible_identity"] != s["physical_identity"]
        
def test_pair_generation():
    pairs = get_pairs()
    assert len(pairs) > 0
    
def test_dataset_loader_example():
    pairs = get_pairs()
    labels_found = set(p["label"] for p in pairs)
    assert "genuine" in labels_found
    assert "different_person" in labels_found
    # Since val/test might not have enough identities for a swap depending on the split size, we just check if any impersonation exists
    assert "impersonation" in labels_found

    # Load one image
    img = cv2.imread(pairs[0]["probe_path"])
    assert img is not None
