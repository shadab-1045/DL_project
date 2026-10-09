# Final V4 Held-Out Test Evaluation (Phase 7F.15)

## 1. Artifact Verification & Provenance
Before test execution, the final model artifacts were securely verified against their established SHA-256 hashes:
- **Phase 7F.11 Visual / Uncalibrated Checkpoint**: `0c5e8463023e2d603070ad4aec435be2bfc16c14115237942e38380983acd192` [PASS]
- **Phase 7F.12 Corrected Identity Checkpoint**: `ca9394dbb3e737e518de4aa77632c3e25b532abce5034e25cfe4a5d11c34837c` [PASS]
- **Phase 7F.13 Final Fusion Checkpoint**: `f2647012bc2863d988e65f980310cac9b60a0beb8c3e068f38b564f4383be2f5` [PASS]

## 2. Dataset Composition
The exact, unmodified `test_pairs_v2.csv` manifest was evaluated:
- Genuine: 387
- Different Person: 90
- Impersonation: 100
- **Total: 577**

## 3. Final V4 Metrics (Held-Out Test Set)
The held-out test results show that the model successfully generalized to unseen data.

- **Overall Accuracy**: 87.87%
- **Macro Precision / Recall / F1**: 0.8430 / 0.8673 / 0.8435
- **Weighted F1**: 0.8805

**Full 3x3 Confusion Matrix (0=Gen, 1=Diff, 2=Imp):**
```text
               Pred_Gen    Pred_Diff    Pred_Imp
Actual_Gen        346         35           6
Actual_Diff         0         88           2
Actual_Imp         22          5          73
```

**Per-Class Metrics (Precision / Recall / F1):**
- **Genuine**: 0.9402 / 0.8941 / 0.9166
- **Different Person**: 0.6875 / 0.9778 / 0.8073
- **Impersonation**: 0.9012 / 0.7300 / 0.8066

## 4. Branch Diagnostics on Test
The isolated branches provided consistent, mathematically bounded signals for the test set inputs:

| Class | Mean P(Identity Match) (Std) | Mean P(Synthetic) (Std) |
|-------|------------------------------|-------------------------|
| **Genuine** | 0.8983 (0.2078) | 0.0305 (0.1096) |
| **Different Person** | 0.1281 (0.1172) | 0.0325 (0.1129) |
| **Impersonation** | 0.8280 (0.2740) | 0.7293 (0.3626) |

## 5. Feature-Neutralization Ablations (Test Set)
To evaluate the contribution of each modality on the test set, we repeated the ablations using the exact neutralization constants derived from the validation phase (`P(Synth) = 0.1294`, `P(Id) = 0.7117`).

- **Normal V4 Test Recall**: Gen=0.894, Diff=0.977, Imp=0.730
- **Visual Signal Neutralized**: Impersonation recall fell to **0.0000** (model identifies 0/100 impersonations).
- **Identity Signal Neutralized**: Different Person recall fell to **0.0000** (model identifies 0/90 different persons).

This provides strong evidence that the final fusion classifier relies on both branches to successfully classify the three conditions.

## 6. Validation vs. Test Generalization
The final V4 architecture generalized highly effectively to the held-out test distribution, demonstrating slightly improved performance over validation:

- **Accuracy**: 0.8094 (Val) → **0.8787** (Test) [+0.0693]
- **Genuine Recall**: 0.8057 (Val) → **0.8941** (Test) [+0.0884]
- **Different Person Recall**: 0.9889 (Val) → **0.9778** (Test) [-0.0111]
- **Impersonation Recall**: 0.6562 (Val) → **0.7300** (Test) [+0.0738]

## 7. Baseline Contextual Comparison
The results of this evaluation can be contextualized against prior model iterations (note that V1 utilized a flawed dataset/preprocessing routine and V2/V3 suffered from early-fusion optimization failures, so direct numerical comparisons require caution):
- **C-Adv V1**: Reached ~76% impersonation recall, but was compromised by identity-leakage in the un-split dataset and lack of distinct impersonation vs different_person target separation.
- **C-Adv V3 (Early Fusion on V2 Data)**: Reached 0% impersonation recall due to severe shortcut learning, where the model entirely ignored the high-dimensional visual tensor in favor of the identity cosine similarity.
- **V4 Final**: Restores and isolates the anti-impersonation signal. By strictly bottlenecking the feature representations into a late-fusion 2D topology, V4 achieves 73% impersonation recall on the strict V2 dataset while simultaneously maintaining 97.7% recall against unmanipulated different persons.

## 8. Reproducibility
- The test inference was executed twice consecutively in the same environment. Both passes yielded strictly identical predictions and metrics across all 577 samples. 
- Sample-level predictions have been saved to `experiments/model_c_v4/test_predictions.csv` (Hash: `5d4d5f4189d520fd5db652834e6a919d9f46754cfbb72d7ae31c448128bbc64e`).

## 9. Limitations
While the V4 late-fusion framework effectively mitigates shortcut learning, the architecture currently outputs a high false-positive rate for `Different Person` classification when presented with `Genuine` samples (35 Genuine samples misclassified as Different Person). This is directly related to the inherent variance in ArcFace cosine similarities extracted from unconstrained images. Future iterations may benefit from multi-frame temporal smoothing or a more sophisticated non-linear identity representation than a pure 1D scalar.

***
*This concludes the final quantitative evaluation. No post-hoc recalibration, threshold tuning, or architecture modifications were applied.*
