> **Correction note (added during review).** Read the headline numbers with these caveats:
> 1. **The 94% "rejection" describes this probe set, not ArcFace's resistance to InSwapper.** In `experiments/final_evaluation/final_decision_results.csv`
>    the 100 impersonation probes have a median ArcFace similarity of **0.105** to the claimed identity (threshold 0.2445); all 100 had a face detected.
>    These swaps barely carry the source identity. On a separate held-out set of 262 LFW / FaceForensics++ InSwapper swaps
>    (`unified-swap-defense/eval_results/system_results.json`) the median similarity is ~0.9 and ArcFace accepts **99.6%**. The script that
>    generated this repository's swap probes (`generate_controlled_swaps.py`) is not in the repository, so the cause of the difference is not established.
> 2. **Usability cost.** Genuine `VERIFIED` falls from 71.6% to 51.4% once Model C is added (see section 5 onward).
> 3. **FGSM only, and a small test set** (100 impersonation probes). The adversarial rows should not be read as general robustness.
> 4. **Model C here is C-Control / C-Adv**, not the `ModelCV4` fusion model used by the live pipeline.

# Phase 5 Final Held-Out Test Evaluation Report

## 1. Executive Summary

This report presents the final evaluation of the Adversarially Robust Face Identity Verification System on the strictly held-out test dataset (`data/manifests/test_pairs.csv`). The evaluation quantifies the system's ability to recognize legitimate identities while rejecting face-swapped impersonations, separating the effects of the identity engine (Model A) and the anti-impersonation module (Model C).

**Key Quantitative Findings:**
1. **Identity Rejection:** The baseline ArcFace identity engine (Model A) rejected 94.0% of the evaluated impersonation probes (routing them to `UNKNOWN`) and accepted 6.0% as the claimed identity at the validation-calibrated threshold.
2. **System Integration (Clean):** Integrating C-Control resulted in a final `VERIFIED` rate of 0.0% for impersonation probes. The 6.0% of impersonations that bypassed Model A were successfully intercepted by Model C and routed to `SUSPECTED_IMPERSONATION`. This security improvement traded off against usability, with the Genuine `VERIFIED` rate dropping from 71.6% to 51.4%.
3. **Adversarial Degradation:** At an $L_\infty$ FGSM perturbation of $\epsilon=0.10$, C-Control's impersonation recall degraded to 0.0.
4. **Robustness Improvement:** Under the same $\epsilon=0.10$ condition, the adversarially trained model (C-Adv) retained an impersonation recall of 0.59.

---

## 2. Model Hashes & Freeze State
Prior to evaluation, all models were frozen to ensure reproducible evaluation metrics.

- **Model A (`w600k_r50.onnx`)**: `4C06341C33C2CA1F86781DAB0E829F88AD5B64BE9FBA56E56BC9EBDEFC619E43`
- **C-Control (`best_model.pt`)**: `1CF70BA964B5E60C6632AC8532A7BA8BCEC9F4509E425CE9CE6DF71E00C86E51`
- **C-Adv (`best_model.pt`)**: `A562487A5B3CC085E3CC1EED6FBEA93748376FB75BBD2FBDF3D0E871F966AFBD`

---

## 3. Test Manifest Verification
The integrity of `data/manifests/test_pairs.csv` was programmatically verified.
- **Total pairs:** 577 (Genuine: 387, Different Person: 90, Impersonation: 100)
- **Identity Overlap:** 0 overlapping reference identities with train/val sets.
- **Leakage:** No duplicate sample IDs or missing files were found.

---

## 4. Model A (Identity Engine) Test Metrics
Model A (ArcFace) threshold was calibrated on the validation set by maximizing balanced accuracy, yielding $\tau = 0.2445$.

Applying this threshold to the test set yielded:
- **Genuine Acceptance Rate (TAR):** 71.58% (Recall: 0.716, Precision: 0.979)
- **False Rejection Rate (FRR):** 28.42%
- **Different Person FAR:** 0.00%
- **Impersonation Acceptance Rate:** 6.00% 

---

## 5. Model C Clean Evaluation ($\epsilon=0.0$)
Evaluating the anti-impersonation module on clean data isolates its baseline classification capabilities.

| Metric | C-Control | C-Adv |
| :--- | :--- | :--- |
| **Accuracy** | 0.671 | 0.659 |
| **Macro F1** | 0.659 | 0.626 |
| **Genuine F1** | 0.732 | 0.732 |
| **Different Person F1** | 0.259 | 0.186 |
| **Impersonation F1** | 0.985 | 0.959 |

---

## 6. Model C Adversarial Evaluation (White-box FGSM)
Structural robustness against adversarial gradients was evaluated using model-specific (white-box) FGSM on the normalized inputs.

| Model | $\epsilon$ | Impersonation Recall | Impersonation F1 |
| :--- | :---: | :---: | :---: |
| C-Control | 0.01 | 0.050 | 0.064 |
| C-Adv | 0.01 | 0.120 | 0.192 |
| C-Control | 0.05 | 0.670 | 0.465 |
| C-Adv | 0.05 | 0.630 | 0.696 |
| C-Control | 0.10 | 0.000 | 0.000 |
| C-Adv | 0.10 | 0.590 | 0.694 |

**Paired Statistical Significance (1,000 Bootstrap/Permutation Iterations):**
A paired randomization test was performed computing `C-Adv - C-Control` differences directly on the paired test samples (Random Seed: 42).
- **Impersonation F1 ($\epsilon=0.10$):** Mean Difference = +0.694. Out of 1,000 permutations, 0 were as or more extreme than the observed difference ($p < 0.001$).
- **Impersonation Recall ($\epsilon=0.10$):** Mean Difference = +0.590. Out of 1,000 permutations, 0 were as or more extreme than the observed difference ($p < 0.001$).

---

## 7. Final Decision State Cross-Tabulation
The complete system logic maps the combinations of Model A and Model C outputs into three final routing states: `VERIFIED`, `UNKNOWN`, and `SUSPECTED_IMPERSONATION`.

| Ground Truth | VERIFIED | UNKNOWN | SUSPECTED_IMPERSONATION |
| :--- | :--- | :--- | :--- |
| **Genuine** (387) | 199 | 188 | 0 |
| **Different Person** (90) | 0 | 90 | 0 |
| **Impersonation** (100) | 0 | 94 | 6 |

**State-Specific Routing Rates:**
- **Genuine $\rightarrow$ VERIFIED:** 51.4%
- **Different Person $\rightarrow$ VERIFIED:** 0.0%
- **Impersonation $\rightarrow$ VERIFIED:** 0.0% (Final Verified Impersonation Rate)

*Note: The `UNKNOWN` state functions as a safe rejection for mismatched identities and ambiguous presentations. It is not semantically identical to `different_person`.*

---

## 8. Impersonation Pipeline Analysis
Tracing the 100 impersonation samples through the logical layers clarifies the distinct roles of the identity engine and the anti-impersonation module:

- **Layer 1 (Identity Engine):** 94 of the 100 impersonation probes were rejected by Model A due to low identity similarity, preempting Model C entirely and resulting in `UNKNOWN`.
- **Layer 2 (Model C):** The remaining 6 impersonation probes successfully bypassed Model A.
- **Layer 3 (Final State):** Model C explicitly classified all 6 of those bypassing probes as `impersonation`. The final decision engine therefore correctly routed these 6 to `SUSPECTED_IMPERSONATION`.

**Final Security Posture:**
- Safe Rejection or Unknown Rate: 100.0%
- Suspected Impersonation Detection Rate: 6.0% (the exact fraction of the threat that reached Model C)
- Final Verified Impersonation Rate: 0.0%

---

## 9. Conclusion
The held-out evaluation confirms that identity-aware synthetic presentation classification (Model C) effectively hardens biometric verification. Among the evaluated impersonation probes, 94.0% were rejected by the identity engine before reaching the VERIFIED state, while the remaining 6.0% were explicitly classified as SUSPECTED_IMPERSONATION by Model C. The complete system allowed 0.0% of impersonations to reach the `VERIFIED` state. Furthermore, adversarial training (C-Adv) provides statistically significant structural defense against white-box gradient attacks, preserving detection capabilities where the undefended baseline fails completely.
