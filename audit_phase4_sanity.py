import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from src.models.model_c import AntiImpersonationModel
from src.data.model_c_dataset import AntiImpersonationDataset
from src.training.adversarial import fgsm_attack

def generate_random_noise(images, epsilon):
    noise = torch.sign(torch.randn_like(images))
    perturbed = images + epsilon * noise
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    min_val = (0.0 - mean) / std
    max_val = (1.0 - mean) / std
    
    return torch.max(torch.min(perturbed, max_val), min_val)

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return tensor * std + mean

def save_qa_image(clean, adv, path):
    # Denormalize to [0, 1]
    clean_denorm = denormalize(clean).squeeze(0).cpu().clamp(0, 1)
    adv_denorm = denormalize(adv).squeeze(0).cpu().clamp(0, 1)
    
    diff = torch.abs(adv_denorm - clean_denorm)
    diff_amp = (diff * 50).clamp(0, 1) # amplify by 50x for visibility
    
    import torchvision.transforms as T
    to_pil = T.ToPILImage()
    
    clean_pil = to_pil(clean_denorm)
    adv_pil = to_pil(adv_denorm)
    diff_pil = to_pil(diff_amp)
    
    # concat side by side
    w, h = clean_pil.size
    combined = Image.new('RGB', (w*3, h))
    combined.paste(clean_pil, (0, 0))
    combined.paste(adv_pil, (w, 0))
    combined.paste(diff_pil, (w*2, 0))
    combined.save(path)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    
    val_ds = AntiImpersonationDataset('data/manifests/val_pairs.csv')
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    c_control = AntiImpersonationModel().to(device)
    c_control.load_state_dict(torch.load('experiments/model_c_control/best_model.pt', map_location=device, weights_only=True))
    c_control.eval()
    
    stats_file = 'experiments/model_c_adv/fgsm_perturbation_stats.csv'
    os.makedirs('experiments/model_c_adv/attack_qa', exist_ok=True)
    
    with open(stats_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', 'epsilon', 'max_normalized_delta', 'mean_abs_normalized_delta', 'max_raw_delta', 'mean_abs_raw_delta'])
    
    diagnostic_file = 'experiments/model_c_adv/random_noise_diagnostic.csv'
    with open(diagnostic_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', 'true_label', 'clean_pred', 'fgsm_pred', 'random_pred'])

    num_samples = 20
    qa_count = 5
    epsilons = [0.01, 0.05, 0.10]
    
    qa_md = open('experiments/model_c_adv/attack_qa/FGSM_VISUAL_QA.md', 'w')
    qa_md.write("# FGSM Visual QA\n\nFor each sample (left to right): Clean Image, Adversarial Image, Amplified Difference (50x)\n\n")
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    for i, (imgs, embs, labels) in enumerate(val_loader):
        if i >= num_samples:
            break
            
        imgs, embs, labels = imgs.to(device), embs.to(device), labels.to(device)
        
        # 1. Evaluate clean
        with torch.no_grad():
            logits_clean = c_control(imgs, embs)
            pred_clean = torch.argmax(logits_clean, dim=1).item()
            
        for eps in epsilons:
            # 2. Evaluate FGSM
            imgs.requires_grad = True
            perturbed_imgs = fgsm_attack(c_control, imgs, embs, labels, criterion, eps)
            with torch.no_grad():
                logits_fgsm = c_control(perturbed_imgs, embs)
                pred_fgsm = torch.argmax(logits_fgsm, dim=1).item()
                
            # Compute Deltas
            norm_delta = torch.abs(perturbed_imgs - imgs).detach()
            
            raw_imgs = denormalize(imgs).detach()
            raw_pert = denormalize(perturbed_imgs).detach()
            raw_delta = torch.abs(raw_pert - raw_imgs) * 255.0
            
            with open(stats_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    i, eps, 
                    norm_delta.max().item(), norm_delta.mean().item(),
                    raw_delta.max().item(), raw_delta.mean().item()
                ])
                
            # 3. Evaluate Random
            if eps == 0.01:
                rand_imgs = generate_random_noise(imgs.detach(), eps)
                with torch.no_grad():
                    logits_rand = c_control(rand_imgs, embs)
                    pred_rand = torch.argmax(logits_rand, dim=1).item()
                    
                with open(diagnostic_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([i, labels.item(), pred_clean, pred_fgsm, pred_rand])
                    
                # Save QA images
                if i < qa_count:
                    path = f'experiments/model_c_adv/attack_qa/sample_{i}_eps_0.01.png'
                    save_qa_image(imgs.detach(), perturbed_imgs.detach(), path)
                    qa_md.write(f"### Sample {i}\n![Sample {i}](../../../{path})\n")

    qa_md.close()
    print("Sanity audit artifacts generated.")
    
    # Let's also print the mathematical verification for sample 0, eps 0.01
    imgs, embs, labels = val_ds[0]
    imgs = imgs.unsqueeze(0).to(device)
    embs = embs.unsqueeze(0).to(device)
    labels = torch.tensor([labels]).to(device)
    
    raw_img = denormalize(imgs) * 255.0
    print("\nMATHEMATICAL VERIFICATION (Sample 0, eps=0.01):")
    print(f"Raw image range (0-255 expected): [{raw_img.min().item():.2f}, {raw_img.max().item():.2f}]")
    print(f"Normalized tensor range: [{imgs.min().item():.4f}, {imgs.max().item():.4f}]")
    
    perturbed = fgsm_attack(c_control, imgs, embs, labels, criterion, 0.01)
    print(f"Perturbed tensor range: [{perturbed.min().item():.4f}, {perturbed.max().item():.4f}]")
    
    diff = torch.abs(perturbed - imgs)
    print(f"Max normalized diff: {diff.max().item():.6f} (Expected <= 0.01)")
    
    raw_diff = torch.abs(denormalize(perturbed) - denormalize(imgs)) * 255.0
    print(f"Max raw diff (8-bit scale): {raw_diff.max().item():.6f}")

if __name__ == "__main__":
    main()
