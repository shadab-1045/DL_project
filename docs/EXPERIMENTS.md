# Experiment Plan

## E1 — Identity baseline
Question: does Model A correctly recognize enrolled identities?

Measure:
- verification accuracy
- similarity distributions
- false acceptance
- false rejection

## E2 — Baseline vulnerability
Feed Model A:
- genuine target identity
- unrelated person
- controlled face-swap impersonation

Record identity similarity and cases where swapped samples resemble the target identity.

## E3 — C-Control
Train Model C architecture without adversarial example generation.

Measure:
- precision
- recall
- macro F1
- ROC-AUC where appropriate
- confusion matrix
- genuine false alarms

## E4 — C-Adv
Same architecture and split, but add hard/adversarial examples.

Compare C-Control vs C-Adv on exactly the same held-out evaluation data.

## E5 — Cross-generator generalization
Where feasible:
- train using one/some manipulation sources
- evaluate on a different generator such as Deep-Live-Cam

## E6 — Post-processing robustness
Evaluate:
- JPEG compression
- resize
- blur
- brightness/contrast
- video compression

## E7 — Identity generalization
Use identities not present in training. Keep identity split before pair generation.

## E8 — Live webcam
Use consenting participants:
1. genuine enrolled user
2. unknown user
3. face-swapped impersonation

Measure live identity and anti-impersonation outputs.

## E9 — Temporal stability
Compare 5/10/15-frame histories or other configured windows.

Measure:
- decision stability
- false transitions
- detection delay

## E10 — Runtime
Measure:
- FPS
- median latency
- p95 latency
- GPU memory
- overall end-to-end latency

## Required artifacts per experiment
```text
experiments/<id>/
  config.yaml
  metrics.json
  predictions.csv
  confusion_matrix.png
  roc_curve.png
  notes.md
  checkpoint_reference.txt
```

## Result discipline
Every reported number must be traceable to:
- experiment ID
- dataset manifest
- checkpoint
- configuration
- threshold
- random seed

No target results are predetermined.
