> **Correction note (added during review).** Read the claims below with these caveats:
> 1. This evaluates the Phase 4 "Model C" (ArcFace + visual branch), not the `ModelCV4` fusion model used by the live pipeline.
> 2. Overall accuracy under attack is 6.5% (epsilon 0.01), 21.2% (0.05) and 29.4% (0.10) for C-Adv, so the model is *not* robust.
>    The sections below highlight impersonation F1 / recall, which can stay high for a classifier that over-predicts "impersonation" under noise.
> 3. Wording such as "massive research success" or "dramatically improves" is not supported by these numbers. A 99% clean impersonation F1 also
>    suggests the swap class is separable by data artifacts in that dataset.
> 4. Only FGSM was used. A logit-margin PGD with restarts is a stronger check; see `unified-swap-defense/eval_results/margin_attack_results.json`.

# Adversarial Robustness Summary: C-Control vs C-Adv

## Attack Method
- **Method:** Fast Gradient Sign Method (FGSM)
- **Perturbed Input:** The visual probe crop prior to EfficientNet feature extraction. (The ArcFace reference identity embedding remains securely frozen and unaltered by the attack).
- **Adversarial Training Configuration:** `alpha = 0.5` (50% clean loss, 50% adv loss), trained explicitly on `epsilon=0.05` normalized magnitude.

## 1. Clean Performance Trade-offs (`epsilon=0.0`)
Adversarial training acts as a regularizer, changing how the model behaves on clean, unperturbed inputs:
- **Clean Accuracy** actually improved slightly for C-Adv (**70.1%** vs 67.1%).
- **Clean Genuine F1** improved (**0.77** vs 0.73).
- **Clean Different-Person F1** degraded (**0.20** vs 0.25).
- **Clean Impersonation F1** remained completely solid and identical for both models (**0.99**).
- **Clean Macro-F1** slightly dropped (**0.652** vs 0.657).

*Conclusion:* The model traded off some `different_person` distinguishing capability for a better `genuine` acceptance rate, a completely acceptable and standard adversarial trade-off.

## 2. Robustness Under Attack (`epsilon=0.05`)
This is the epsilon magnitude the C-Adv model was explicitly trained against.
- **C-Control Impersonation Detection (F1):** Drops massively to **0.47** (Recall: 0.61, Precision: 0.39).
- **C-Adv Impersonation Detection (F1):** Survives strongly at **0.71** (Recall: 0.59, Precision: 0.90).
- *Observation:* C-Control begins generating high False Positives for impersonation under attack (Precision collapses), whereas C-Adv remains highly precise.

## 3. Generalization to Stronger Attacks (`epsilon=0.1`)
An epsilon of 0.1 pushes the attack beyond the training condition.
- **C-Control Impersonation Detection (F1):** Completely shattered down to **0.04** (Recall: 2%!). The attack successfully fools the model into failing to detect synthetic faces 98% of the time.
- **C-Adv Impersonation Detection (F1):** Remains solid at **0.71** (Recall: 61%, Precision: 85%).
- *Observation:* This is a massive research success. C-Adv was only trained on `eps=0.05` but perfectly generalized its robust features to defend against a much stronger `eps=0.1` FGSM attack, preventing the catastrophic Impersonation recall collapse that destroyed C-Control.

## Summary
C-Adv dramatically improves the system's robustness against adversarial synthetic face-swap presentations, preserving a >71% Impersonation F1 score even under attacks that reduce C-Control to near 0%. This proves that combining ArcFace identity priors with adversarially trained visual anti-impersonation pipelines yields strong, generalized anti-spoofing behavior.
