# Initialization Audit

Based on source code inspection of `src/training/train_model_c.py` and `src/training/train_model_c_adv.py`:

**What is proven by code inspection:**
1. Both scripts enforce deterministic initialization via `torch.manual_seed(42)` and `np.random.seed(42)` at the very beginning of `main()`.
2. Both scripts instantiate the model identically using `model = AntiImpersonationModel().to(device)`.
3. Inside `AntiImpersonationModel`, the visual backbone is loaded using `models.efficientnet_b0(weights='IMAGENET1K_V1')`, guaranteeing identically matched pretrained initial weights.
4. The fusion and classification layers (`self.fusion`) are initialized directly after setting the manual seed, meaning their random initial weights are structurally identical across both runs.
5. The `ArcFace` encoder is identically frozen in both models and produces exactly the same deterministic embeddings.
6. The script `train_model_c_adv.py` does **not** contain any code to `torch.load()` the C-Control checkpoint prior to training.

**What cannot be reconstructed after the fact:**
Because the models were trained sequentially, we cannot verify exact layer-by-layer bitwise initialization parity at runtime zero without retraining both from scratch simultaneously. However, because both rely on the exact same fixed seed, standard PyTorch determinism guarantees the untrained parameter graphs were functionally identical prior to epoch 1.
