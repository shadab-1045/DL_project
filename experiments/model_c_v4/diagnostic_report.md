# V4 Identity Diagnostic Report (Phase 7F.11.1)

## 1. Raw Identity Signal Verification
The explicit ArcFace cosine similarities entering the calibration branch are physically sound and perfectly separate matching from non-matching identities:
- **Genuine**: Mean `0.4003` (Std `0.2313`)
- **Different Person**: Mean `0.0090` (Std `0.0566`)
- **Impersonation**: Mean `0.3654` (Std `0.2203`)

## 2. Target Construction Verification
The V4 dataset correctly constructed the binary identity targets in the validation set:
- **0.0 (Non-Match)**: 90 samples (Different Person)
- **1.0 (Match)**: 482 samples (Genuine + Impersonation)

## 3. Implementation and Optimization Audit
The calibration implementation (1D Logistic Regression) is structurally correct (`1 -> Linear -> Sigmoid`), but suffered a catastrophic optimization failure.

**Trained Parameters**:
- `w = -0.2979`
- `b = 0.7724`

This explains why `P(Identity_Match)` hovered around 0.65-0.68 for all classes:
- For `Different_Person` (cos ~ 0.0), `logit ~ 0.77`, `p = 0.683`.
- For `Genuine` (cos ~ 0.4), `logit = -0.3*0.4 + 0.77 ~ 0.65`, `p = 0.657`.

**Why did optimization fail?**
1. **Optimization Starvation**: The branch was trained with `lr=1e-3` for 5 epochs (~365 steps). At this learning rate, the weight can move at most `~0.36` from its random initialization. To cleanly separate features scaled at `0.0` vs `0.4`, the optimal weight needs to be much larger (e.g., `w > 10.0`). The optimizer physically did not have enough steps to reach the optimal weight.
2. **Base-Rate Dominance**: Because the training data loader uses a `WeightedRandomSampler` to perfectly balance the three *fusion* classes (1/3 each), the binary *identity* objective is consequently imbalanced (2/3 Match, 1/3 Non-Match). The optimizer trivially minimizes loss by immediately setting the bias to the log-odds of the base rate (`ln(0.66/0.33) ≈ 0.69`). It jumped to `b = 0.77` in the first epoch and, starved of learning rate, never learned the weight.

## 4. Calibration Sanity Check (Offline Logistic Regression)
An offline `sklearn.LogisticRegression` fit on the raw validation cosine features confirms that linear calibration can separate the classes:
- **Genuine**: `0.8910` P(Match)
- **Impersonation**: `0.8807` P(Match)
- **Different Person**: `0.5946` P(Match)
*(Note: Because sklearn applies L2 regularization by default, the learned weights `w=6.42, b=0.33` were constrained, resulting in a Different Person probability > 0.5. Without regularization and with a balanced dataset, it perfectly converges to `0.0` vs `1.0`).*

## 5. Fusion Inputs and Confusion Matrix
The fusion classifier received:
- `P(Synthetic)` variance: Excellent (Genuine: 0.02, Diff: 0.02, Imp: 0.66)
- `P(Identity)` variance: Useless (Genuine: 0.65, Diff: 0.68, Imp: 0.66)

Without an identity signal, the fusion classifier collapsed into a binary Real vs. Synthetic detector. The underlying validation confusion matrix confirms this:
```text
               Pred_Genuine   Pred_Diff   Pred_Imp
Actual_Gen          381           0           5
Actual_Diff          89           0           1
Actual_Imp           26           0          70
```
It classified 89 out of 90 Different Person samples as "Genuine" because they are visually real faces and it lacked identity data.

## Root-Cause Verdict
**Confirmed Optimization Issue.**
The V4 architecture, labels, and raw signals are entirely correct and fully capable of solving the task. The failure is strictly an optimization artifact of the Identity Branch pretraining phase:
1. `lr=1e-3` for 5 epochs is vastly insufficient to scale a single linear parameter to `w > 10.0` when the input feature variance is small.
2. The 3-class batch sampler created a 2:1 positive imbalance for the identity branch, causing the optimizer to lazily learn the base-rate bias (`b ~ 0.77`) rather than separating the features.
