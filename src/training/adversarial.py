import torch

def fgsm_attack(model, images, embeddings, labels, criterion, epsilon):
    """
    Generates an FGSM adversarial example.
    
    Args:
        model: PyTorch model.
        images: Clean visual probe inputs [B, 3, H, W].
        embeddings: Reference ArcFace embeddings [B, D].
        labels: True labels for the inputs.
        criterion: Loss function.
        epsilon: Maximum perturbation magnitude in the normalized input space.
        
    Returns:
        perturbed_images: Adversarially perturbed images.
    """
    if epsilon == 0:
        return images.detach()
        
    # We only perturb the images
    images = images.clone().detach().to(images.device)
    images.requires_grad = True
    
    embeddings = embeddings.detach()
    labels = labels.to(images.device)
    
    # Forward pass
    logits = model(images, embeddings)
    
    # Calculate loss
    loss = criterion(logits, labels)
    
    # Zero all existing gradients
    model.zero_grad()
    
    # Calculate gradients of model in backward pass
    loss.backward()
    
    # Collect data_grad
    data_grad = images.grad.data
    
    # Create the perturbed image by adjusting each pixel of the input image
    sign_data_grad = data_grad.sign()
    
    # Perturb the image
    perturbed_images = images + epsilon * sign_data_grad
    
    # Standard ImageNet normalization bounds
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    
    min_val = (0.0 - mean) / std
    max_val = (1.0 - mean) / std
    
    perturbed_images = torch.max(torch.min(perturbed_images, max_val), min_val)
    
    return perturbed_images.detach()
