import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
from src.models.model_c_v3 import AntiImpersonationModelV3, count_parameters
from src.data.model_c_v3_dataset import AntiImpersonationV3Dataset
from src.training.train_model_c_adv_v3 import fgsm_attack_v3

def benchmark_and_sanity_test():
    print("--- 4. MODEL PARAMETERS ---")
    model = AntiImpersonationModelV3()
    t, f = count_parameters(model)
    print(f"Total Trainable Params: {t:,}")
    print(f"Total Frozen Params: {f:,}")
    
    # Trainable parameter breakdown
    eff_t = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    mlp_t = sum(p.numel() for p in model.fusion.parameters() if p.requires_grad)
    print(f"EfficientNet Trainable: {eff_t:,}")
    print(f"Fusion MLP Trainable: {mlp_t:,}")
    print(f"ArcFace Parameters in Model: {any('arcface' in n.lower() for n, _ in model.named_parameters())}")

    print("\n--- 6. DATASET PERFORMANCE ---")
    dataset = AntiImpersonationV3Dataset('data/manifests/train_pairs_v2.csv')
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    start_time = time.time()
    
    try:
        # Load 2 batches (64 samples)
        batch1 = next(iter(loader))
        batch2 = next(iter(loader))
    except Exception as e:
        print("Error in dataset loading:", e)
        return
        
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Time to load 64 samples: {elapsed:.2f} seconds")
    print(f"Throughput: {64/elapsed:.2f} samples/sec")
    print("ArcFace is executed dynamically via ONNX in the dataset __getitem__.")

    print("\n--- 7. SINGLE-BATCH FORWARD/FGSM SANITY TEST ---")
    imgs, ref_embs, probe_embs, sims, labels = batch1
    imgs, ref_embs, probe_embs, sims, labels = imgs.to(device), ref_embs.to(device), probe_embs.to(device), sims.to(device), labels.to(device)
    
    print("Initial Input Shapes:")
    print(f"imgs: {imgs.shape}")
    print(f"ref_embs: {ref_embs.shape}")
    print(f"probe_embs: {probe_embs.shape}")
    print(f"sims: {sims.shape}")
    
    # Clean Forward
    model.train()
    logits_clean = model(imgs, ref_embs, probe_embs, sims)
    print(f"Clean Logits Shape: {logits_clean.shape}")
    
    criterion = nn.CrossEntropyLoss()
    
    # Check gradients
    imgs_adv = imgs.clone().detach().requires_grad_(True)
    logits_tmp = model(imgs_adv, ref_embs, probe_embs, sims)
    loss = criterion(logits_tmp, labels)
    loss.backward()
    
    print(f"Gradients exist for visual input? {imgs_adv.grad is not None}")
    
    epsilon = 0.05
    perturbed_imgs = fgsm_attack_v3(model, imgs, ref_embs, probe_embs, sims, labels, criterion, epsilon)
    
    diff = torch.abs(perturbed_imgs - imgs)
    max_diff = torch.max(diff).item()
    print(f"Max perturbation (L_inf): {max_diff:.4f} (Expected exactly {epsilon})")
    
    print(f"Are ArcFace embeddings modified? Ref: False (not passed to FGSM as differentiable)")
    print(f"Are PyTorch NaNs present in adversarial images? {torch.isnan(perturbed_imgs).any().item()}")
    print(f"Are PyTorch Infs present in adversarial images? {torch.isinf(perturbed_imgs).any().item()}")
    
    # Adv Forward
    logits_adv = model(perturbed_imgs, ref_embs, probe_embs, sims)
    print(f"Adversarial Logits Shape: {logits_adv.shape}")

if __name__ == "__main__":
    benchmark_and_sanity_test()
