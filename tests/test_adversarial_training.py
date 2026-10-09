import torch
import torch.nn as nn
import pytest
from src.training.adversarial import fgsm_attack

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 16 * 16 + 512, 3)
        
    def forward(self, images, embeddings):
        x = self.conv(images)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, embeddings], dim=1)
        return self.fc(x)

@pytest.fixture
def setup_fgsm():
    model = DummyModel()
    model.eval()
    
    batch_size = 4
    images = torch.rand((batch_size, 3, 16, 16))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = (images - mean) / std

    
    embeddings = torch.rand((batch_size, 512))
    labels = torch.randint(0, 3, (batch_size,))
    criterion = nn.CrossEntropyLoss()
    
    return model, images, embeddings, labels, criterion

def test_fgsm_epsilon_zero(setup_fgsm):
    model, images, embeddings, labels, criterion = setup_fgsm
    perturbed = fgsm_attack(model, images, embeddings, labels, criterion, epsilon=0.0)
    assert torch.allclose(images, perturbed)

def test_fgsm_epsilon_positive_changes_input(setup_fgsm):
    model, images, embeddings, labels, criterion = setup_fgsm
    perturbed = fgsm_attack(model, images, embeddings, labels, criterion, epsilon=0.1)
    assert not torch.allclose(images, perturbed)

def test_fgsm_perturbation_magnitude(setup_fgsm):
    model, images, embeddings, labels, criterion = setup_fgsm
    epsilon = 0.1
    perturbed = fgsm_attack(model, images, embeddings, labels, criterion, epsilon=epsilon)
    diff = torch.abs(perturbed - images)
    assert torch.max(diff) <= epsilon + 1e-5

def test_fgsm_reference_embedding_unchanged(setup_fgsm):
    model, images, embeddings, labels, criterion = setup_fgsm
    embeddings_before = embeddings.clone()
    _ = fgsm_attack(model, images, embeddings, labels, criterion, epsilon=0.1)
    assert torch.allclose(embeddings, embeddings_before)

def test_fgsm_model_parameters_unchanged(setup_fgsm):
    model, images, embeddings, labels, criterion = setup_fgsm
    params_before = [p.clone() for p in model.parameters()]
    _ = fgsm_attack(model, images, embeddings, labels, criterion, epsilon=0.1)
    params_after = [p.clone() for p in model.parameters()]
    for p_before, p_after in zip(params_before, params_after):
        assert torch.allclose(p_before, p_after)

def test_fgsm_labels_unchanged(setup_fgsm):
    model, images, embeddings, labels, criterion = setup_fgsm
    labels_before = labels.clone()
    _ = fgsm_attack(model, images, embeddings, labels, criterion, epsilon=0.1)
    assert torch.all(labels == labels_before)

def test_fgsm_batch_dimensions(setup_fgsm):
    model, images, embeddings, labels, criterion = setup_fgsm
    perturbed = fgsm_attack(model, images, embeddings, labels, criterion, epsilon=0.1)
    assert perturbed.shape == images.shape

def test_fgsm_no_nan_inf(setup_fgsm):
    model, images, embeddings, labels, criterion = setup_fgsm
    perturbed = fgsm_attack(model, images, embeddings, labels, criterion, epsilon=0.1)
    assert not torch.isnan(perturbed).any()
    assert not torch.isinf(perturbed).any()

def test_fgsm_only_on_training_data_in_script():
    # Inspect the source code to prove it doesn't attack validation set during training
    import ast
    with open('src/training/train_model_c_adv.py', 'r') as f:
        source = f.read()
    
    assert "fgsm_attack" in source
    assert "perturbed_imgs = fgsm_attack" in source
    # Ensure fgsm_attack is not called in the validation block
    # By checking that fgsm_attack is strictly inside the train_loader loop
    
    # We can do a string search: the word fgsm_attack does not appear after val_loader loop starts
    val_loop_idx = source.find("for imgs, embs, labels in tqdm(val_loader")
    fgsm_call_idx = source.find("fgsm_attack(")
    last_fgsm_call_idx = source.rfind("fgsm_attack(")
    
    assert val_loop_idx != -1
    assert fgsm_call_idx != -1
    assert last_fgsm_call_idx < val_loop_idx


