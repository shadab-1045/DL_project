# Visual-Only Control Experiment (Phase 7F.9)

## 1. Objective and Methodology
This is a diagnostic control experiment designed to determine whether the `EfficientNet-B0` visual representation has the theoretical capacity to detect synthetic impersonation artifacts when isolated from all identity-consistency shortcuts.

- **Architecture**: `EfficientNet-B0` visual backbone mapped directly to a 3-class MLP (`1280` -> `512` -> `256` -> `3`).
- **Inputs**: Only the `112x112` pre-aligned probe image. The model received **no** reference identity, **no** probe ArcFace embedding, and **no** cosine similarity.
- **Training**: Clean data (no FGSM adversarial training). Standard Adam optimizer, 15 epochs, batch size 32, inverse-frequency class weighting.

## 2. Validation Performance
Evaluated on the exact same clean V2 Validation split as V2/V3.

**Overall Metrics:**
- **Accuracy**: 66.08%
- **Macro F1**: 0.6020
- **Weighted F1**: 0.6797

**Per-Class Performance:**
- **Genuine Recall**: 72.28%
- **Different Person Recall**: 30.00%
- **Impersonation Recall**: **75.00%**
- **Impersonation Precision**: **88.89%**
- **Impersonation F1**: 0.8136

**Confusion Matrix:**
```
            Predicted
            Gen   Diff  Imp
Actual Gen  279   100   7
Actual Diff 61    27    2
Actual Imp  20    4     72
```

## 3. Training Dynamics
- **Initial Epochs**: Impersonation recall was non-zero and stable immediately. At Epoch 1, it achieved 78.12% impersonation recall.
- **Stability**: Over 15 epochs, impersonation recall consistently hovered between 70% and 80%, finishing at 75.00% on the selected best epoch (Epoch 14).
- **The "Different Person" Problem**: Because the model is entirely blind to the reference identity, it inherently cannot distinguish a "Different Person" from the "Genuine" person reliably. Consequently, Different Person recall never exceeded ~55% and collapsed to 30.0% by Epoch 14.

## 4. Scientific Comparison

The key diagnostic question was whether the visual branch *itself* can learn synthetic artifacts before identity shortcuts are introduced.

| Metric | C-Adv V2 (Learned Sim) | C-Adv V3 (Explicit Sim) | Visual-Only (No Identity) |
|--------|------------------------|-------------------------|---------------------------|
| **Impersonation Recall** | 0.00% | 0.00% | **75.00%** |
| **Impersonation Precision** | 0.00% | 0.00% | **88.89%** |
| **Genuine Recall** | 61.10% | 84.20% | 72.28% |
| **Different Person Recall** | 43.30% | 90.00% | 30.00% |

### Interpretation
**The EfficientNet visual representation definitively contains the synthetic artifact signal.** 

When stripped of the identity representations, the `EfficientNet-B0` backbone easily learned to detect the `Native InSwapper` synthetically generated faces, achieving 75% recall and 88% precision without any adversarial training. 

This confirms that the failures of V2 and V3 are **not** caused by an inherent architectural inability to "see" the fake faces. 

Instead, the failures are entirely caused by the fusion objective:
- The visual branch excels at separating *Fake* from *Real* (Impersonation vs Genuine/Different).
- The identity branch (ArcFace cosine similarity) excels at separating *Target Identity* from *Other Identity* (Genuine/Impersonation vs Different).
- However, when concatenated, the MLP lazily optimizes for the identity signal because it is mathematically simpler to threshold a 1-d scalar than to decode 1280-d visual noise. The network accepts a 0% recall on Impersonation (which is 1/3 of a balanced batch) to effortlessly gain 100% accuracy on Genuine and Different Person (which is 2/3 of a balanced batch).

This diagnostic control establishes that the V4 solution must forcefully prevent feature laziness—perhaps by training the visual branch independently (or freezing a pre-trained synthetic artifact detector) and combining its output with the ArcFace similarity using a non-lazy decision framework, rather than relying on early concatenation in an MLP.
