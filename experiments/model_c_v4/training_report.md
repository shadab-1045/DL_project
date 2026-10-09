# V4 Training Report (Phase 7F.11)

## 1. Architecture and Parameter Counts
The V4 architecture employs strict independent branch pretraining and completely bottlenecked 2-dimensional late fusion.
- **Visual Branch (Binary: Real vs Synthetic)**: `EfficientNet-B0` backbone (pretrained weights) + MLP head (`1280 -> 512 -> 1`). Output is `P(Synthetic)`.
- **Identity Branch (Binary: Match vs Non-Match)**: 1D Logistic Regression (1 weight, 1 bias) on the explicit ArcFace cosine similarity. Output is `P(Identity_Match)`.
- **Fusion Branch (3-Class)**: Small MLP (`2 -> 16 -> 3`).

**Trainable Parameters**:
- Visual Branch: ~4M (EfficientNet-B0) + ~655K (Head)
- Identity Branch: 2 parameters
- Fusion Branch: 99 parameters

## 2. Integrity-Test Results
Before training, `scratch/test_model_c_v4.py` was executed to verify the structural integrity of the V4 methodology:
- `[PASS]` Class mappings and targets: Genuine (Real, Match), Different_Person (Real, Non-Match), Impersonation (Synthetic, Match) accurately produced correct binary labels.
- `[PASS]` Forward pass dimensional constraints: Fusion input was strictly asserted to be `(batch, 2)`.
- `[PASS]` Frozen branch gradient constraints: Verified that backpropagating from the fusion loss resulted in `None` gradients for the visual and identity branch parameters.

## 3. Training Protocol
The model was trained strictly on `train_pairs_v2.csv` (seed=42) in three independent phases:
- **Step 1 (Visual Branch)**: 10 Epochs, Adam `lr=1e-4`, BCE Loss. 
- **Step 2 (Identity Calibration)**: 5 Epochs, Adam `lr=1e-3`, BCE Loss.
- **Step 3 (Frozen Late Fusion)**: 5 Epochs, Adam `lr=1e-3`, Cross-Entropy Loss. Both Visual and Identity branches were completely frozen (`requires_grad=False`).

## 4. Evaluation and Validation Control Results
Evaluated on `val_pairs_v2.csv` only. **The V2 test set was strictly excluded and never evaluated.**

### Branch Diagnostics (Binary Objectives)
- **Visual Branch Accuracy**: 93.71%
- **Identity Branch Accuracy**: 84.27%

**Mean Predicted Probabilities by Actual Class**:
- **Genuine**: `P(Synthetic)` = 0.021 | `P(Identity_Match)` = 0.658
- **Different Person**: `P(Synthetic)` = 0.026 | `P(Identity_Match)` = 0.683
- **Impersonation**: `P(Synthetic)` = 0.662 | `P(Identity_Match)` = 0.660

### 2D Late Fusion (Final 3-Class Model)
- **Overall Accuracy**: 78.85%
- **Genuine Recall**: 98.70%
- **Different Person Recall**: 0.00%
- **Impersonation Recall**: 72.92%
- **Impersonation Precision**: 92.11%

### Feature Neutralization Ablations
To isolate the contribution of each branch in the fusion layer, we replaced the branch's output with its dataset mean `P`:
- **Neutralize Visual** `[Mean P(Synthetic), Actual P(Identity)]`: Impersonation Recall collapsed to **0.00%**, Genuine Recall rose to 100%. The model defaulted to "Genuine".
- **Neutralize Identity** `[Actual P(Synthetic), Mean P(Identity)]`: Impersonation Recall remained **72.92%**, Genuine Recall remained **98.70%**, Different Person Recall remained **0.00%**. 

## 5. Scientific Interpretation and Anomalies
The **Visual Branch** successfully learned to detect synthetic manipulation, achieving an excellent separation (P(Synthetic) is ~0.66 for Impersonation and ~0.02 for real faces). Because it was trained independently and bottlenecked before fusion, the fusion layer had no choice but to utilize this signal, correctly rescuing the **Impersonation Recall to 72.92%** (compared to 0% in V2 and V3). 

However, an unexpected anomaly occurred in the **Identity Branch**. The 1D Logistic Regression failed to confidently separate Genuine from Different Person based on cosine similarity, predicting `P(Identity_Match) ~ 0.65 - 0.68` across all three classes. Because the identity input provided no separating information to the fusion layer, the fusion layer learned to rely *entirely* on the Visual Branch, effectively functioning as a binary real vs. synthetic classifier. Consequently, the model correctly detected Genuine and Impersonation but entirely failed to identify Different Person (Recall = 0.00%), misclassifying them as Genuine.

This indicates that while the architecture successfully preserved the visual anti-spoofing signal, the specific optimization or scale of the 1D identity calibration layer failed during training.

## 6. Artifacts
- **Files Created**: `src/models/model_c_v4.py`, `src/data/model_c_v4_dataset.py`, `scratch/test_model_c_v4.py`, `src/training/train_model_c_adv_v4.py`, `src/training/eval_model_c_adv_v4.py`.
- **Checkpoint Hash (SHA-256)**: Saved in `experiments/model_c_v4/checkpoint_hash.txt` (`5c7a52e00e004...` format).
- **V2 Test Set**: Untouched. No live demo integration performed.
