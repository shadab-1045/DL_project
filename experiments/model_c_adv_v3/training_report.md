# Model C V3 Training Report

## 1. Experimental Setup
- **Architecture**: `Model C V3` (EfficientNet-B0 + explicitly frozen ArcFace probe/reference embeddings + explicit cosine similarity).
- **Initialization**: Fresh random initialization (seed 42). 
- **Training Data**: Corrected V2 Train Manifest (1161 Genuine, 870 Different Person, 297 Impersonation).
- **Validation Data**: Corrected V2 Val Manifest (386 Genuine, 90 Different Person, 96 Impersonation).
- **Hyperparameters**: Adam (lr=1e-4), Batch Size 32, 15 Epochs, Alpha 0.5.
- **Adversarial Attack**: FGSM (Epsilon 0.05). Perturbations applied *only* to the visual backbone; identity representations were mathematically decoupled and remained fixed.
- **Checkpoint Criterion**: Maximum Validation Macro F1 (Epoch 7 selected).

## 2. Validation Performance (Clean / Eps = 0.0)

**Overall Metrics:**
- **Accuracy**: 70.98%
- **Macro F1**: 0.4880

**Per-Class Recall:**
- **Genuine**: 84.20%
- **Different Person**: 90.00%
- **Impersonation**: 0.00%

## 3. Robustness Evaluation

| Epsilon | Accuracy | Macro F1 | Impersonation Precision | Impersonation Recall | Impersonation F1 |
|---------|----------|----------|-------------------------|----------------------|------------------|
| 0.00    | 70.98%   | 0.4880   | 0.0000                  | 0.0000               | 0.0000           |
| 0.01    | 43.53%   | 0.2711   | 0.2000                  | 0.0417               | 0.0690           |
| 0.05    | 50.17%   | 0.4482   | 0.2576                  | 0.3542               | 0.2982           |
| 0.10    | 62.06%   | 0.4263   | 0.0000                  | 0.0000               | 0.0000           |

## 4. Scientific Comparison: V1 vs V2 vs V3

The key scientific question was: *Does explicitly supplying pretrained ArcFace identity consistency allow the classifier to exploit the visual synthetic-artifact information that V2 failed to use?*

| Metric | C-Adv V1 (Unaligned) | C-Adv V2 (Learned Sim) | C-Adv V3 (Explicit Sim) |
|--------|----------------------|------------------------|-------------------------|
| **Impersonation Recall** | ~99.0% (Artifact Leakage) | 0.00% | 0.00% |
| **Impersonation Precision**| ~98.0% | 0.00% | 0.00% |
| **Impersonation F1** | ~0.98 | 0.00 | 0.00 |
| **Genuine Recall** | ~95.0% | 61.10% | 84.20% |
| **Different Person Recall** | ~90.0% | 43.30% | 90.00% |
| **Macro F1** | ~0.95 | 0.2970 | 0.4880 |

### Interpretation & Hypothesis
Providing the explicit pretrained ArcFace identity consistency completely rescued the model's ability to distinguish **Genuine** from **Different Person**. In V2, Genuine recall was 61% and Different Person recall was 43%. In V3, because the model is handed the explicit ArcFace cosine similarity, Genuine recall shot up to 84% and Different Person recall to 90%.

However, **Impersonation Recall remains at 0.00%**. 

**Why did V3 fail to use the visual synthetic-artifact information?**
The network exhibited "shortcut learning" (or feature laziness). The explicit `cosine_similarity` feature is an incredibly strong, perfectly stable 1-dimensional signal that effortlessly separates *Different Person* from *[Genuine + Impersonation]*. 

To separate *Genuine* from *Impersonation*, the MLP must learn to decode the complex, noisy 1280-dimensional visual feature map to find subtle synthetic artifacts. Instead of learning to fuse these modalities, the MLP simply learned to threshold the explicit cosine similarity metric:
- If `cosine_similarity > threshold` ➔ Predict `Genuine` (misclassifying all Impersonations).
- If `cosine_similarity < threshold` ➔ Predict `Different_Person`.

Because the visual features were effectively ignored in favor of the single scalar identity metric, the FGSM adversarial attack—which only perturbed the visual input—did not impose a sufficient loss penalty to force the network to rely on the visual features. The model accepted the training loss penalty on the Impersonation class (which makes up 33% of the balanced batch) in exchange for near-perfect, easy accuracy on the Genuine and Different Person classes (66% of the batch).

## 5. Reproducibility
- The checkpoint exists at `experiments/model_c_adv_v3/best_model.pt`.
- V1 and V2 artifacts and checkpoints remain completely untouched.
- The V2 test set remains strictly un-evaluated.
