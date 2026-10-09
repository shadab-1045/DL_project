# Test Integrity Audit

**Goal:** Prove `data/manifests/test_pairs.csv` was strictly held out until Phase 5, completely isolated from any training or validation decisions.

### File Contamination Search
- In `src/training/train_model_c.py` and `train_model_c_adv.py`, the test dataset is **never loaded**. The validation loader is explicitly fed `val_pairs.csv`. The code operates entirely unaware of `test_pairs.csv`.
- In Phase 4, the robustness evaluations in `audit_phase4_robustness.py` strictly parsed `val_pairs.csv` to calculate the robustness degradation.
- No model threshold, adversarial strength (alpha, epsilon), early stopping trigger, or checkpoint selection mechanism (`best_model.pt` saved at Epoch 5) had any access to the test manifest.

### Architectural & Preprocessing Integrity
- `src/data/model_c_dataset.py` dynamically normalizes images independently on-the-fly. No dataset-level aggregate statistics were calculated over the test set that could cause data leakage into the training phase (we used standard ImageNet means/stds for EfficientNet and pretrained ArcFace).
- `test_pairs.csv` uniquely contains identity-disjoint samples guaranteed by `src/data/build_identity_split.py`. The reference identities enrolled in the test set never appeared in the training set.

### Phase 5 Strict Hold-out Execution
- The `calibrate_threshold` step inside `src/evaluation/run_final_evaluation.py` mathematically locks the threshold value by maximizing balanced accuracy on `val_pairs.csv` exclusively.
- The derived threshold is blindly applied to the TEST set.
- All evaluation results generated herein are final. No models were retrained based on observing these final test scores.
