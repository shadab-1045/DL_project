# Model Specification

## Model A — Baseline Identity Recognizer

Purpose: identity recognition only.

Implementation: pretrained InsightFace/ArcFace model. Start with an approved pretrained model such as an InsightFace model-zoo checkpoint.

Training:
- no real/fake classifier training
- no anti-impersonation training
- identity enrollment only

Runtime:
```text
face → ArcFace embedding → gallery similarity → identity/unknown
```

## Model C — Proposed Anti-Impersonation System

Purpose:
Given a current presentation and the embedding of the claimed/best-matching identity, determine:
- genuine
- different_person
- impersonation

### Architecture
```text
face/context crop
      ↓
visual backbone (EfficientNet-B0 or Xception)
      ↓
visual feature
         +
ArcFace identity embedding
      ↓
fusion
      ↓
MLP / classification head
      ↓
3 classes
```

### Initial implementation
- ArcFace frozen.
- Anti-impersonation branch trainable.
- Model implemented in PyTorch.
- Input dimensions configurable.
- Architecture selected through configuration.

## C-Control
Same Model C architecture and dataset split, but ordinary supervised training only.

## C-Adv
Same architecture, with hard/adversarial examples in training.

## Output
Raw logits:
```text
[genuine, different_person, impersonation]
```

Convert to probabilities with softmax.

## Loss
Initial:
```text
L_clean = CrossEntropy(clean_logits, labels)
```

Adversarial:
```text
L_total = L_clean + lambda_adv * L_adv
```

All hyperparameters configurable.

## Class balancing
Record class counts. Use class weights or balanced sampling only when justified and documented.

## Checkpoints
Store:
- weights
- config
- seed
- dataset manifest reference
- software commit
- validation metrics

## Runtime interface
```python
predict(face_crop, identity_embedding) -> {
    "genuine": float,
    "different_person": float,
    "impersonation": float
}
```

Do not let model inference itself decide the final security state; the decision engine handles thresholds and temporal smoothing.
