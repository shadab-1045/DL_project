# Model C V3 Architecture

## 1. Motivation
In `C-Adv V2`, the model attempted to learn a cross-modal projection from its `EfficientNet-B0` visual features (1280-d) to the `ArcFace` reference identity space (512-d). The purpose of this projection was to dynamically compute an internal cosine similarity score that would distinguish Genuine presentations from Different Person presentations. 

However, Phase 7F.5 diagnostics proved that learning this visual→ArcFace projection from scratch on only 30 training identities caused massive overfitting. The internal similarity metric completely collapsed to ~0.01 for all unseen validation identities, starving the final classification MLP of the critical identity-consistency signal.

## 2. V3 Architecture
Model C V3 completely removes the trainable visual→ArcFace projection. Instead, it explicitly leverages the *pretrained* ArcFace identity signal directly within the architecture, bypassing the representation bottleneck.

### Input
- Aligned 112x112 probe face crop.

### Branch A: Visual (Trainable)
- **Backbone**: `EfficientNet-B0`
- **Output**: 1280-dimensional visual feature tensor capturing raw spatial/visual artifacts.

### Branch B: Probe Identity (Frozen)
- **Engine**: Pretrained `buffalo_l` InsightFace ONNX Recognition Model (`w600k_r50.onnx`).
- **Output**: 512-dimensional, L2-normalized probe identity embedding extracted from the exact same aligned face crop.

### Reference Identity (Frozen)
- The existing precomputed 512-dimensional, L2-normalized ArcFace reference embedding for the enrolled identity.

### Explicit Identity Consistency
- The exact scalar cosine similarity between the reference embedding and the probe embedding is computed explicitly prior to fusion: `dot(reference, probe)`.

## 3. Fusion and Classification
The final classifier is an MLP receiving all modalities explicitly concatenated:

**Fusion Input Tensor Dimensions:**
1. Visual Feature: `1280`
2. Reference ArcFace Embedding: `512`
3. Probe ArcFace Embedding: `512`
4. Cosine Similarity: `1`

**Total Input Dimension**: `2305`

The fusion MLP transforms the `2305` features through identical hidden layers (`512 -> 256 -> 3`) to output logits for the three exact classes:
1. `genuine` (Index 0)
2. `different_person` (Index 1)
3. `impersonation` (Index 2)

## 4. Preprocessing & Isolation
- The PyTorch `Dataset` loads the aligned image and immediately extracts the ArcFace probe embedding using ONNX Runtime. This guarantees the visual backbone and the identity backbone evaluate the exact same geometric frame.
- Because ArcFace executes as an ONNX black-box inside the dataset `__getitem__`, it is mathematically insulated from PyTorch autograd. When Adversarial FGSM perturbs the visual input tensor during V3 training, it will only corrupt the `EfficientNet` visual branch, maintaining the explicit instruction to perturb the *visual* input while leaving the identity representation robustly decoupled.
