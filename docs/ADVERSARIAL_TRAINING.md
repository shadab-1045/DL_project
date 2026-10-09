# Adversarial Training Specification

## Objective
Improve robustness of the identity-aware anti-impersonation model against difficult synthetic identity presentations.

## Attack taxonomy

### 1. Semantic attack
Face swap:
```text
physical person B
+
identity/face of A
→ A-on-B impersonation
```

### 2. Hard manipulation variants
Examples:
- JPEG compression
- resizing
- mild blur
- brightness shift
- contrast shift
- color degradation
- video compression

These remain semantically `impersonation`.

### 3. FGSM
Generate model-aware input perturbations:
```text
x_adv = x + epsilon * sign(gradient_x(loss))
```

The perturbation must be constrained and must preserve the semantic sample label.

### 4. PGD
Optional advanced experiment after FGSM is stable.

## Training comparison

### C-Control
Clean + standard face-swap/different-person training.

### C-Adv
Same base training + hard manipulation variants + FGSM examples.

Keep architecture and evaluation fixed.

## Progressive implementation
1. Clean control.
2. Add hard face-swap transformations.
3. Add FGSM.
4. Optional PGD.

Do not begin with PGD.

## Labels
FGSM/PGD versions of an impersonation example remain `impersonation`.

## Configurable parameters
- epsilon
- PGD steps
- step size
- clipping bounds
- clean/adversarial ratio
- lambda_adv
- random seed

## Evaluation
Compare C-Control and C-Adv on:
- genuine test
- different-person test
- normal FaceSwap
- hard FaceSwap
- FGSM test
- optional PGD test
- unseen generator
- Deep-Live-Cam live samples

Never tune thresholds on final test data.

## Interpretation
Do not assume adversarial training always improves every metric. Report trade-offs, especially genuine-user false alarms.
