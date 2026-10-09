import torch
from src.models.model_c_v4 import ModelCV4
from src.data.model_c_v4_dataset import AntiImpersonationV4Dataset

def run_tests():
    print("Running V4 Integrity Tests...")
    ds = AntiImpersonationV4Dataset('data/manifests/val_pairs_v2.csv')
    
    # 1. Test class mappings
    for pair in ds.pairs:
        l = pair['label']
        v = ds.visual_map[l]
        i = ds.identity_match_map[l]
        f = ds.class_map[l]
        
        if l == 'genuine':
            assert v == 0.0, "Genuine visual should be Real (0)"
            assert i == 1.0, "Genuine identity should be Match (1)"
            assert f == 0
        elif l == 'different_person':
            assert v == 0.0, "Different Person visual should be Real (0)"
            assert i == 0.0, "Different Person identity should be Non-Match (0)"
            assert f == 1
        elif l == 'impersonation':
            assert v == 1.0, "Impersonation visual should be Synthetic (1)"
            assert i == 1.0, "Impersonation identity should be Match (1)"
            assert f == 2
            
    print("[PASS] Class mappings and targets")
            
    model = ModelCV4()
    
    # 2. Test dimensionality constraints
    dummy_img = torch.randn(2, 3, 224, 224)
    dummy_sim = torch.randn(2, 1)
    
    fusion_logits, p_synth, p_id = model(dummy_img, dummy_sim)
    
    assert p_synth.shape == (2, 1)
    assert p_id.shape == (2, 1)
    assert fusion_logits.shape == (2, 3)
    print("[PASS] Forward pass dimensional constraints")
    
    # 3. Test parameter frozen state
    for p in model.visual_branch.parameters():
        p.requires_grad = False
    for p in model.identity_branch.parameters():
        p.requires_grad = False
        
    optimizer = torch.optim.Adam(model.fusion.parameters(), lr=1e-3)
    optimizer.zero_grad()
    
    loss = fusion_logits.sum()
    loss.backward()
    
    has_vis_grad = any(p.grad is not None for p in model.visual_branch.parameters())
    has_id_grad = any(p.grad is not None for p in model.identity_branch.parameters())
    has_fus_grad = any(p.grad is not None for p in model.fusion.parameters())
    
    assert not has_vis_grad, "Visual branch received gradient during fusion!"
    assert not has_id_grad, "Identity branch received gradient during fusion!"
    assert has_fus_grad, "Fusion branch did not receive gradient!"
    print("[PASS] Frozen branch gradient constraints")
    
if __name__ == "__main__":
    run_tests()
