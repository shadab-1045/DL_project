# V4 Final 2D Late Fusion Report (Phase 7F.13)

## 1. Scientific Objective
The objective of this phase was to answer a core research question: *Can the two independently learned low-dimensional signals—identity match and synthetic manipulation—be combined to distinguish genuine, different-person, and synthetic impersonation without succumbing to shortcut learning?*

## 2. Methodology & Integrity Constraints
The final `Model C V4` architecture strictly enforced independent modality compression before fusion:
1. **Visual Branch**: Frozen `EfficientNet-B0` pre-trained to output `P(Synthetic)`.
2. **Identity Branch**: Frozen logistic calibration pre-trained to output `P(Identity_Match)`.
3. **Fusion Classifier**: A small 3-class MLP (`2 -> 16 -> 3`) trained exclusively on the concatenated 2D vector `[P(Synthetic), P(Identity_Match)]`.

**Integrity Checks (Passed)**: 
A programmatic unit test ran before and after training, guaranteeing that visual and identity weights were not updated, and that the fusion input was strictly 2-dimensional. The V2 Test Set was strictly isolated and not evaluated.

## 3. Branch Diagnostic Table
The independent branches successfully compressed their inputs into mathematically sound, highly semantic scalar signals:

| Class | Mean P(Id Match) | Mean P(Synthetic) |
|---|---|---|
| **Genuine** | 0.8227 | 0.0209 |
| **Different Person** | 0.1217 | 0.0257 |
| **Impersonation** | 0.8185 | 0.6624 |

## 4. 2D Late Fusion Validation Performance
The Fusion MLP successfully learned to combine the 2D signals into the final 3-class decision:

- **Accuracy**: 80.94%
- **Genuine Recall**: 80.57%
- **Different Person Recall**: 98.89%
- **Impersonation Recall**: 65.62%
- **Impersonation Precision**: 96.92%

**Validation Confusion Matrix**:
```text
               Pred_Gen    Pred_Diff    Pred_Imp
Actual_Gen        311         74           1
Actual_Diff         0         89           1
Actual_Imp         25          8          63
```

*Note: The model successfully recovers 65.6% of synthetic impersonations (up from 0% in V3) and 98.9% of different physical persons, proving that the fusion layer successfully navigates both sub-problems simultaneously.*

## 5. Feature-Neutralization Ablations
To conclusively prove that the fusion layer relies on both modalities, we systematically neutralized each branch by replacing its output with the dataset mean.

### Ablation A: Neutralized Visual Signal (P(Synth) = 0.1294)
- Genuine Recall: 80.83%
- Different Person Recall: 97.78%
- **Impersonation Recall**: **0.00%**
*Expected Degradation Confirmed: Without the visual synthetic artifact signal, the model assumes all identity-matches are Genuine, failing to catch a single impersonation attack.*

### Ablation B: Neutralized Identity Signal (P(Id_Match) = 0.7117)
- Genuine Recall: 99.74%
- **Different Person Recall**: **0.00%**
- Impersonation Recall: 67.71%
*Expected Degradation Confirmed: Without the identity signal, the model assumes all unmanipulated images are Genuine, failing to catch a single physical different-person.*

## 6. Final Verdict
**Success.** 

The V4 controlled Late-Fusion architecture elegantly avoids the specific feature laziness failure mode observed in early-fusion (V3). By isolating the sub-problems and bottlenecking the modalities to 2-dimensional semantic probabilities (`P(Synthetic)` and `P(Identity_Match)`), the fusion layer provides strong evidence that both branches contribute to the final decision and respect the visual branch's anti-spoofing capabilities. 

This completes the required V4 diagnostic and training iterations. No live-demo integrations or held-out test evaluations were performed during this phase.
