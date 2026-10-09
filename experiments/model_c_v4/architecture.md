# Model C V4: Late-Fusion Architecture

## 1. Motivation
The sequence of V2, V3, and Visual-Only experiments isolated a fundamental architectural flaw in early-fusion neural networks for identity-aware anti-spoofing: feature laziness.

In V3, the model concatenated the 1280-d visual feature map with the 1-d explicit ArcFace cosine similarity. Because the 1-d cosine similarity perfectly separates *Different Person* presentations from *Genuine/Impersonation* presentations, it acted as a strong mathematical shortcut. The optimizer minimized loss by thresholding the 1-d cosine scalar, resulting in 0% Impersonation recall.

However, the Phase 7F.9 Visual-Only control showed that the `EfficientNet-B0` visual representation detects Native InSwapper artifacts (75% recall) when isolated from the identity shortcut.

V4 addresses this by utilizing independent branch pretraining with frozen-branch late fusion. The visual and identity branches are trained to independently compress their modalities into low-dimensional probabilities before they are allowed to interact. This prevents the identity signal from bypassing or suppressing the visual synthetic-artifact detector.

## 2. The Visual Branch (Binary: Real vs. Synthetic)

### Binary Formulation
The visual branch is formulated as a Binary Classifier (`Real` vs `Synthetic`), with targets:
- `genuine` → `0` (Real / non-synthetic)
- `different_person` → `0` (Real / non-synthetic)
- `impersonation` → `1` (Synthetic)

**Why Binary?**
Without access to a reference embedding, looking at a single unmanipulated face crop provides no information about whether the subject is the "enrolled user" or a "different person". Both are pristine human faces. A 3-class objective creates label noise for the visual optimizer. By mapping Genuine/Different Person to `Real` and Impersonation to `Synthetic`, the supervision isolates the intended subproblem of detecting synthetic artifacts (though it does not guarantee which specific visual cues the model will ultimately exploit).

### Visual Architecture
- **Input**: Aligned 112x112 probe image → PyTorch `[3, 224, 224]`.
- **Backbone**: `EfficientNet-B0`.
- **Head**: `1280` → `512` → `1` (Sigmoid).
- **Output**: `P(Synthetic)` — The scalar probability that the image contains synthetic face-swap manipulation.

## 3. The Identity Branch

The identity branch must remain interpretable and avoid overparameterization. We will use a lightweight, trainable 1D Logistic Regression to calibrate the explicit ArcFace cosine similarity into `P(Identity_Match)`.

### Identity Formulation
`P(Identity_Match)` is explicitly defined as: whether the *visible face* in the probe matches the enrolled/reference identity, **not** whether the physical person behind the image is the enrolled person.

Therefore, the identity-calibration targets are:
- `genuine` → `identity_match = 1`
- `different_person` → `identity_match = 0`
- `impersonation` → `identity_match = 1`

*(Why is impersonation an identity match? Because the synthetic visible face is intentionally generated to be the enrolled identity, even though the physical person is different).*

### Identity Architecture
- **Input**: Precomputed 512-d ArcFace reference embedding, Precomputed 512-d ArcFace probe embedding.
- **Processing**: Explicit Cosine Similarity: `dot(reference, probe)`.
- **Calibration Head**: A simple 1-dimensional trainable logistic regression (Weight + Bias + Sigmoid).
- **Output**: `P(Identity_Match)`

## 4. Late Fusion Mechanism

The fusion layer receives ONLY the low-dimensional semantic outputs from the two isolated branches.

- **Fusion Input Dimension**: `2` (`[P(Synthetic), P(Identity_Match)]`).
- **Expected Semantic Pattern**:
  - `genuine` = identity match (`1`) + non-synthetic (`0`)
  - `different_person` = identity non-match (`0`) + non-synthetic (`0`)
  - `impersonation` = identity match (`1`) + synthetic (`1`)
- **Fusion Architecture**: A minimal MLP (e.g., `2` → `16` → `3`) or a deterministic rule-based decision engine.
- **Fusion Output**: 3 logits mapping exactly to the semantic classes: `0 = genuine`, `1 = different_person`, `2 = impersonation`.

## 5. Training Strategy: Independent Branch Pretraining with Frozen-Branch Late Fusion

To prevent the identity branch from suppressing the visual branch gradients, V4 employs independent branch pretraining with frozen-branch late fusion:

1. **Step 1: Train the Visual Branch**: Train the visual branch entirely independently on the binary `Real vs Synthetic` objective using Binary Cross Entropy (BCE). 
2. **Step 2: Calibrate the Identity Branch**: Train the 1D logistic regression independently to predict `Identity_Match` on the binary targets.
3. **Step 3: Train Late Fusion**: Freeze both branches. Extract `P(Synthetic)` and `P(Identity_Match)` for the entire dataset. Train the small `2->16->3` fusion MLP on these 2D vectors using 3-class Cross Entropy. 

**Strict Data Separation**:
All branch pretraining and fusion training must use ONLY the V2 training split. The V2 validation split is strictly reserved for model selection/calibration. The V2 test split must remain entirely untouched.

## 6. Trainable vs. Frozen Components
- **Pretrained ArcFace Models**: Strictly FROZEN.
- **Visual Branch**: Trainable during Step 1. Frozen during Step 3.
- **Identity Calibration**: Trainable during Step 2. Frozen during Step 3.
- **Late Fusion MLP**: Trainable during Step 3.

## 7. Prevention of Shortcut Learning
Shortcut learning is heavily discouraged in the V4 fusion stage. The fusion classifier never sees the raw 1280-d visual feature; it only sees `P(Synthetic)`. By independently pretraining the visual branch, we force `P(Synthetic)` to be highly accurate. If the fusion classifier attempts to ignore `P(Synthetic)`, it will suffer massive loss on Impersonation samples, and it cannot bypass `P(Synthetic)` to find another shortcut because no other features are provided.

## 8. Required V4 Validation Controls and Evaluation Plan

V4 must make it possible to report separately:
- **identity-only performance**
- **visual-only performance**
- **two-dimensional late fusion performance**
- **feature-neutralization/ablation** showing the effect of removing or neutralizing each of the two fusion inputs on the final 3-class prediction.

Additionally, V4 must output a required **per-class branch diagnostic table** containing:
- identity-match rate/probability by genuine/different_person/impersonation
- synthetic probability by genuine/different_person/impersonation

## 9. Expected Scientific Interpretation

- **If Fused Impersonation Recall is High**: This supports the hypothesis. It indicates that the visual branch captured synthetic artifacts, the identity branch captured identity, and late fusion combined them without one overriding the other.
- **If Fused Impersonation Recall is Low (but Visual-Only is High)**: This indicates a failure of the Fusion MLP itself, suggesting that the 2D decision boundary is non-linearly complex or that the dataset distribution heavily biases the fusion optimizer. 
- **If Visual-Only Performance is Low**: This indicates the visual branch failed to learn the binary task under the V4 regime, contradicting the Phase 7F.9 control and requiring further dataset or backbone investigation.
