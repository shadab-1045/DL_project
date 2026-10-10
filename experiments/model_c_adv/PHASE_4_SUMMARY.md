> **Correction note (added during review).** Read the claims below with these caveats:
> 1. **Accuracy under attack is low.** `robustness_matrix.csv` (n = 572) gives C-Adv an overall accuracy of 6.5% / 21.2% / 29.4% at
>    FGSM epsilon 0.01 / 0.05 / 0.10 (C-Control: 0.9% / 18.7% / 18.7%). The "61% impersonation recall" below is one class's recall; it can stay
>    high for a classifier that over-predicts "impersonation" under noise, so it is not evidence of robustness on its own.
> 2. **"Perfectly generalizes" and "proof of the research hypothesis" are not supported** by those numbers.
> 3. **Only FGSM was used.** A stronger logit-margin PGD with restarts, applied later to a separate swap detector
>    (`unified-swap-defense/eval_results/margin_attack_results.json`), showed that FGSM/BCE-PGD robustness numbers can be gradient-masking artifacts.
>    C-Adv was not re-tested with it.
> 4. **This is not the live model.** C-Adv (`src/models/model_c.py`) is not the model used by the live pipeline, which loads `ModelCV4`.

# Phase 4 Summary: Adversarial Training (C-Adv)

This phase successfully augmented the baseline Identity-Aware Anti-Impersonation system (`C-Control`) into an adversarially robust model (`C-Adv`). By training the model against dynamically generated Fast Gradient Sign Method (FGSM) attacks, the system learned to detect synthetic impersonation attempts even when an attacker injects mathematically optimized, sub-pixel visual noise designed to fool the network.

Here is the complete file-by-file breakdown of the research and engineering work completed during Phase 4.

## 1. Core Algorithm Implementation

### `src/training/adversarial.py`
- **Purpose:** Implements the core FGSM algorithm.
- **Details:** Contains the `fgsm_attack()` function. It accepts the visual face crop and the frozen ArcFace reference embedding. It computes the gradient of the loss specifically with respect to the visual input tensor, calculates the sign of the gradient, applies the `epsilon` multiplier, and precisely clamps the output back to standard ImageNet normalization bounds to prevent mathematically impossible pixel values. 

## 2. Unit Testing & Mathematical Verification

### `tests/test_adversarial_training.py`
- **Purpose:** Ensures the FGSM attack logic is mathematically flawless and physically isolated.
- **Details:** Contains 9 strict unit tests. It verifies that:
  - `epsilon = 0` returns the exact input tensor.
  - `epsilon > 0` changes the tensor but strictly obeys the $L_\infty$ magnitude bound.
  - The frozen ArcFace identity embedding is completely ignored by the attack.
  - No model parameters or running statistics (like BatchNorm) are inadvertently altered during the attack generation.
  - The training loop physically isolates the attack from contaminating validation data.

## 3. Training Pipeline

### `src/training/train_model_c_adv.py`
- **Purpose:** The adversarial training loop for C-Adv.
- **Details:** 
  - Initializes a fresh `AntiImpersonationModel` from scratch, utilizing the exact same random seeds and ImageNet priors as C-Control (ensuring a fair, identical starting state).
  - Implements the adversarial defense strategy using a 50/50 blended loss formulation: `total_loss = 0.5 * clean_loss + 0.5 * adv_loss`.
  - The model was adversarially trained against `epsilon = 0.05`.
  - Evaluates strictly on *clean* validation data to select the best checkpoint (Epoch 5) to prevent over-tuning on adversarial metrics.

### `run_adv_training.py`
- **Purpose:** Training execution wrapper.
- **Details:** A script that invokes the training loop, capturing standard output, system hardware information, and runtime hyperparameters to store permanently in `experiments/model_c_adv/training_log.txt`.

## 4. Evaluation & Integrity Audits

### `audit_phase4_robustness.py`
- **Purpose:** The comprehensive, white-box evaluation pipeline.
- **Details:** Evaluates both `C-Control` and `C-Adv` against `epsilon` strengths of `0.0` (Clean), `0.01`, `0.05`, and `0.10`. Crucially, this script enforces a strict *white-box* environment: it dynamically generates attacks against C-Control using C-Control's own gradients, and attacks against C-Adv using C-Adv's gradients, ensuring no cross-model contamination. It exports the exhaustive per-class metrics to CSVs.

### `audit_phase4_sanity.py`
- **Purpose:** The ultimate mathematical sanity checker for the results.
- **Details:** 
  - Proved that an `epsilon = 0.01` perturbation corresponds to a microscopic, invisible shift of less than **0.6 pixel intensity** on an 8-bit [0-255] scale.
  - Diagnosed that `C-Control` collapses from 67.1% accuracy to 0.9% accuracy under this invisible perturbation.
  - Ran a Random-Noise diagnostic to prove that the collapse was completely unique to directed FGSM gradients, mathematically verifying that `C-Control` has a severe, genuine vulnerability to adversarial spoofing.

## 5. Artifacts and Results

### Checkpoints and Configs
- **`experiments/model_c_adv/best_model.pt`**: The serialized weights of the completed C-Adv model.
- **`experiments/model_c_adv/checkpoint_metadata.json`**: Hyperparameters, hash validation for data inputs, and selection metrics.

### Result Matrices
- **`experiments/model_c_adv/robustness_matrix.csv`** & **`robustness_report.csv`**: The tabular proof of the research hypothesis. 
  - **The Result:** Under a heavy `epsilon=0.10` attack, `C-Control`'s ability to detect synthetic impersonations is completely shattered (Recall drops to **2%**). Conversely, `C-Adv` perfectly generalizes its defense to survive the attack, maintaining a **61%** Impersonation Recall and a highly stable F1 score of **0.71**.

### Audit Reports
- **`experiments/model_c_adv/initialization_audit.md`**: Proves both models began from identical starting conditions.
- **`experiments/model_c_adv/white_box_audit.md`**: Proves no transfer/black-box attacks were utilized.
- **`experiments/model_c_adv/robustness_summary.md`**: The final thesis interpretation of the metrics, concluding that adversarial training successfully shields the visual fusion layer from adversarial impersonation attacks without destroying clean-data performance.
