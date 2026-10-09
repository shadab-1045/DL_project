import os
import pytest
import pandas as pd
import torch
from src.models.model_c import AntiImpersonationModel
from src.data.model_c_dataset import AntiImpersonationDataset

def test_model_output_shape():
    model = AntiImpersonationModel()
    dummy_img = torch.randn(2, 3, 224, 224)
    dummy_id = torch.randn(2, 512)
    out = model(dummy_img, dummy_id)
    assert out.shape == (2, 3), "Model output must contain exactly 3 logits"

def test_training_labels_valid():
    df = pd.read_csv('data/manifests/train_pairs.csv')
    valid_labels = {"genuine", "different_person", "impersonation"}
    assert set(df['label'].unique()).issubset(valid_labels), "Invalid labels found in training set"

def test_validation_labels_valid():
    df = pd.read_csv('data/manifests/val_pairs.csv')
    valid_labels = {"genuine", "different_person", "impersonation"}
    assert set(df['label'].unique()).issubset(valid_labels), "Invalid labels found in validation set"

def test_impersonation_identity_collision():
    df = pd.read_csv('data/manifests/train_pairs.csv')
    imp = df[df['label'] == 'impersonation']
    collisions = imp[imp['visible_identity'] == imp['physical_identity']]
    assert len(collisions) == 0, "Source/target identity collision found for an impersonation row"

def test_missing_probe_image(tmp_path):
    # Create fake CSV with missing image path
    fake_csv = tmp_path / "fake.csv"
    fake_csv.write_text("pair_id,probe_path,reference_embedding_path,label\n1,missing.jpg,gallery/1/embedding.npy,genuine")
    
    dataset = AntiImpersonationDataset(str(fake_csv))
    with pytest.raises(FileNotFoundError):
        _ = dataset[0]

def test_missing_reference_embedding(tmp_path):
    # Create fake CSV with missing embedding path but valid probe path
    probe_img = tmp_path / "probe.jpg"
    probe_img.write_text("fake image data") # Not a real image but it will fail at the embedding check first if we swap order, or we can make a real PIL image
    
    from PIL import Image
    im = Image.new('RGB', (10, 10))
    im.save(str(probe_img))
    
    fake_csv = tmp_path / "fake.csv"
    fake_csv.write_text(f"pair_id,probe_path,reference_embedding_path,label\n1,{str(probe_img)},missing.npy,genuine")
    
    dataset = AntiImpersonationDataset(str(fake_csv))
    with pytest.raises(FileNotFoundError):
        _ = dataset[0]

def test_invalid_label(tmp_path):
    probe_img = tmp_path / "probe.jpg"
    from PIL import Image
    im = Image.new('RGB', (10, 10))
    im.save(str(probe_img))
    
    emb_path = tmp_path / "emb.npy"
    import numpy as np
    np.save(str(emb_path), np.random.randn(512))
    
    fake_csv = tmp_path / "fake.csv"
    fake_csv.write_text(f"pair_id,probe_path,reference_embedding_path,label\n1,{str(probe_img)},{str(emb_path)},fake_label")
    
    dataset = AntiImpersonationDataset(str(fake_csv))
    with pytest.raises(ValueError):
        _ = dataset[0]

def test_identity_leakage():
    train_df = pd.read_csv('data/manifests/train_pairs.csv')
    val_df = pd.read_csv('data/manifests/val_pairs.csv')
    test_df = pd.read_csv('data/manifests/test_pairs.csv')
    
    train_ids = set(train_df['reference_identity']).union(set(train_df['physical_identity']))
    val_ids = set(val_df['reference_identity']).union(set(val_df['physical_identity']))
    test_ids = set(test_df['reference_identity']).union(set(test_df['physical_identity']))
    
    assert len(train_ids.intersection(val_ids)) == 0, "Leakage between train and val"
    assert len(train_ids.intersection(test_ids)) == 0, "Leakage between train and test"
    assert len(val_ids.intersection(test_ids)) == 0, "Leakage between val and test"
