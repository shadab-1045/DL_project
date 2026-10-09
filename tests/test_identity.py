import os
import numpy as np
import pytest
from src.identity.arcface import IdentityEngine
from src.identity.gallery import IdentityGallery

@pytest.fixture
def engine():
    # Force CPU for tests to ensure it runs anywhere
    return IdentityEngine(model_name="buffalo_l", device="cpu")

@pytest.fixture
def gallery(tmp_path):
    gallery_dir = tmp_path / "test_gallery"
    return IdentityGallery(gallery_dir=str(gallery_dir))

def test_embedding_shape_and_normalization(engine):
    # Dummy image (no face, so should return None, but let's test similarity)
    # Since we can't guarantee a face in a random array, we'll test the similarity function directly.
    emb1 = np.random.rand(512).astype(np.float32)
    emb1 /= np.linalg.norm(emb1)
    
    assert emb1.shape == (512,)
    assert np.isclose(np.linalg.norm(emb1), 1.0)

def test_cosine_similarity(engine):
    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([0.0, 1.0, 0.0])
    emb3 = np.array([1.0, 0.0, 0.0])
    
    sim_diff = engine.compute_similarity(emb1, emb2)
    sim_same = engine.compute_similarity(emb1, emb3)
    
    assert np.isclose(sim_diff, 0.0)
    assert np.isclose(sim_same, 1.0)

def test_gallery_save_load(gallery):
    emb = np.random.rand(512).astype(np.float32)
    emb /= np.linalg.norm(emb)
    
    gallery.save_identity("TestUser", emb)
    
    gallery2 = IdentityGallery(gallery_dir=gallery.gallery_dir)
    loaded_emb = gallery2.identities["TestUser"]
    
    assert np.allclose(emb, loaded_emb)

def test_unknown_threshold(engine, gallery):
    # Save a known embedding
    known_emb = np.array([1.0, 0.0, 0.0])
    gallery.save_identity("KnownUser", known_emb)
    
    # Test with similar embedding
    similar_emb = np.array([0.9, 0.1, 0.0])
    similar_emb /= np.linalg.norm(similar_emb)
    res_match = gallery.identify(engine, similar_emb, threshold=0.8)
    assert res_match["status"] == "VERIFIED"
    assert res_match["identity"] == "KnownUser"
    
    # Test with different embedding
    diff_emb = np.array([0.0, 1.0, 0.0])
    res_unknown = gallery.identify(engine, diff_emb, threshold=0.8)
    assert res_unknown["status"] == "UNKNOWN"
    assert res_unknown["identity"] is None

def test_malformed_input(engine, gallery):
    res = gallery.identify(engine, None)
    assert res["status"] == "NO_FACE"
