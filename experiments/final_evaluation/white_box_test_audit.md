# White-Box FGSM Test Audit

This document formally verifies that the adversarial attacks (FGSM) evaluated on the test set were strictly model-specific (white-box) and not transfer attacks.

## Verification
- In `experiments/final_evaluation/run_final_evaluation.py` (and the `paired_bootstrap.py` script), the target model undergoing evaluation is passed directly into the `fgsm_attack` function:
  ```python
  imgs = fgsm_attack(model, imgs, embs, labels, criterion, epsilon)
  ```
- This ensures that:
  - When evaluating **C-Control**, `fgsm_attack` generates adversarial gradients by backpropagating through **C-Control**'s weights.
  - When evaluating **C-Adv**, `fgsm_attack` generates adversarial gradients by backpropagating through **C-Adv**'s weights.

## Conclusion
The attacks generated represent true white-box conditions. No transfer vulnerabilities were mixed into the primary robustness metric. C-Adv's performance at $\epsilon=0.10$ is measured against noise structurally optimized to defeat C-Adv itself.
