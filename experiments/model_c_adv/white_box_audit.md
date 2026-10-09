# White-Box Audit

An explicit audit of `audit_phase4_robustness.py` proves this is a mathematically fair, true white-box evaluation.

The evaluation function is defined as:
```python
def evaluate_model_robustness(model, val_loader, device, criterion, epsilon):
    ...
    if epsilon > 0:
        perturbed_imgs = fgsm_attack(model, imgs, embs, labels, criterion, epsilon)
```

In the main execution block, this function is called strictly independently for each model instance:
```python
acc, prec, rec, f1, supp = evaluate_model_robustness(c_control, val_loader, device, criterion, eps)
...
acc, prec, rec, f1, supp = evaluate_model_robustness(c_adv, val_loader, device, criterion, eps)
```

Because `c_control` is explicitly passed as the `model` argument to `fgsm_attack` during its evaluation row, the gradients used to synthesize the adversarial perturbations are derived directly and exclusively from the C-Control parameters. 

Likewise, when evaluating C-Adv, `c_adv` is passed to `fgsm_attack`, meaning the perturbations are built solely using C-Adv's learned parameters.

**Conclusion:** No cross-model adversarial examples (black-box transfer) were used. The evaluation strictly assesses each model's vulnerability against an attacker possessing perfect, white-box knowledge of that specific model's gradients.
