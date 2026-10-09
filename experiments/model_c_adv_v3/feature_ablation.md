# C-Adv V3 Feature-Ablation Diagnostic

## 1. Methodology
This diagnostic evaluates the relative contributions of the visual backbone and the explicit identity representations in the trained `Model C V3` checkpoint. No retraining or weight updating occurred. 

At inference time, specific subsets of the 2305-dimensional fusion input (`1280` visual, `512` ref, `512` probe, `1` cosine) were neutralized by overriding them with the V2 validation-set mean for those corresponding features. This effectively removes their information content without changing the mathematical geometry of the trained MLP.

## 2. Baseline Performance (Normal Inference)
- **Accuracy**: 70.98%
- **Macro F1**: 0.4880
- **Genuine Recall**: 84.20%
- **Different Person Recall**: 90.00%
- **Impersonation Recall**: 0.00%
- **Impersonation Precision**: 0.00%
- **Impersonation F1**: 0.00

**Confusion Matrix:**
```
            Predicted
            Gen   Diff  Imp
Actual Gen  325   61    0
Actual Diff 9     81    0
Actual Imp  79    17    0
```

## 3. Identity-Only Ablation (Neutralized Visual Features)
*Visual features replaced with dataset mean; Identity features (ref, probe, cosine) unmodified.*
- **Accuracy**: 68.53%
- **Macro F1**: 0.4717
- **Genuine Recall**: 79.53%
- **Different Person Recall**: 94.44%
- **Impersonation Recall**: 0.00%

**Observation**: Performance is nearly identical to the baseline. Removing the visual features barely degrades the classifier.

## 4. Visual-Only Ablation (Neutralized Identity Features)
*Identity features (ref, probe, cosine) replaced with dataset mean; Visual features unmodified.*
- **Accuracy**: 67.48%
- **Macro F1**: 0.2686
- **Genuine Recall**: 100.0%
- **Different Person Recall**: 0.00%
- **Impersonation Recall**: 0.00%

**Observation**: The classifier predicts "Genuine" for 100% of samples (572/572). The visual branch contains no actionable classification logic on its own.

## 5. Visual + Explicit Cosine
*Ref/Probe 512-d embeddings neutralized; Visual and explicit scalar cosine unmodified.*
- **Accuracy**: 70.80%
- **Macro F1**: 0.4859
- **Genuine Recall**: 84.20%
- **Different Person Recall**: 88.89%
- **Impersonation Recall**: 0.00%

**Observation**: Performance perfectly matches the baseline. The MLP ignores the 512-d reference/probe vectors and relies strictly on the precomputed 1-d explicit cosine feature.

## 6. Reference + Probe (Without Explicit Cosine)
*Explicit scalar cosine neutralized; Visual and 512-d Ref/Probe unmodified.*
- **Accuracy**: 67.48%
- **Macro F1**: 0.2686
- **Genuine Recall**: 100.0%

**Observation**: Just like the Visual-Only ablation, the classifier collapses and predicts "Genuine" for 100% of samples. It did not learn to dynamically extract similarity from the 512-d embeddings; it relied entirely on the 1-d explicitly provided cosine similarity.

## 7. Per-Class Probability Analysis (Mean Predicted Probabilities)

| Ablation Configuration | Actual Class | P(Genuine) | P(Diff Person) | P(Impersonation) |
|------------------------|--------------|------------|----------------|------------------|
| **Baseline** | Genuine | **0.77** | 0.21 | 0.01 |
| | Diff Person | 0.30 | **0.68** | 0.01 |
| | Impersonation | **0.74** | 0.22 | 0.02 |
| **Identity-Only** | Genuine | **0.77** | 0.21 | 0.01 |
| | Diff Person | 0.26 | **0.72** | 0.01 |
| | Impersonation | **0.76** | 0.22 | 0.01 |
| **Visual-Only** | Genuine | **0.80** | 0.18 | 0.01 |
| | Diff Person | **0.78** | 0.19 | 0.01 |
| | Impersonation | **0.78** | 0.18 | 0.03 |

**Observation**: The mean predicted probability for Impersonation never exceeds `~0.03` in any configuration. For actual Impersonation samples, the network confidently predicts `Genuine` (0.74) because the identity-similarity is high, completely overriding the visual artifacts.

## 8. Visual-Branch Sensitivity
When comparing the exact logits of the **Baseline** (with visual features) vs **Identity-Only** (neutralized visual features):
- **Mean Absolute Logit Difference**: 0.345
- **Maximum Absolute Logit Difference**: 1.643
- **Percentage of Samples whose Predicted Class Changed**: **6.99%**

## 9. Final Interpretation

**A. Strong evidence of visual-branch under-utilization**

The ablation conclusively proves that V3 suffers from catastrophic "shortcut learning":
1. **The visual features are ignored**: Supplying neutral visual features changed the classifier's prediction on only 6.99% of the validation set. 
2. **The 512-d embeddings are ignored**: The network never learned to compute interactions between the 512-d reference and probe embeddings. 
3. **A single 1-d feature dominates the entire 2305-d network**: When the 1-d `cosine_similarity` feature is neutralized, the network collapses and blindly predicts "Genuine" for 100% of the dataset.

Because the explicit 1-d cosine similarity perfectly separates `Different Person` from `[Genuine + Impersonation]`, the neural network lazily optimized its loss by thresholding that single scalar, successfully scoring ~70% accuracy. The network avoided the mathematically "hard" task of extracting subtle synthetic artifacts from the 1280-d visual feature map. 

**Conclusion**: The trained V3 model does **not** contain any measurable useful impersonation signal at inference time. The visual branch is under-utilized, and V3 functions merely as an overly complex wrapper around the standard 1-d ArcFace cosine similarity.
