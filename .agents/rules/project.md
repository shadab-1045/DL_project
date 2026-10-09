# Persistent Project Engineering Rules

## Project definition
- The main task is identity-aware anti-impersonation.
- Do not turn it into a generic REAL-vs-FAKE classifier.
- Model A is identity-only ArcFace.
- Model C is ArcFace + trainable anti-impersonation branch.
- C-Control is an internal research ablation; C-Adv is the proposed adversarial version.

## Research integrity
- Never invent metrics or claim success without measurements.
- Compare C-Control and C-Adv on identical evaluation protocols.
- Split identities/videos before pair generation.
- Never leak frames from the same video across splits.
- Calibrate thresholds on validation only.
- Save seeds, configs, manifests, checkpoints, and metrics.

## Data
- Keep raw, processed, generated, adversarial, and live data separate.
- Validate source/target/swap metadata.
- Do not use live demo users for training.
- Respect dataset licensing/access requirements.

## Third-party software
- Integrate stable projects rather than rewriting them.
- Keep Deep-Live-Cam changes minimal.
- Prefer in-memory processed-frame integration.
- Document any third-party patch.

## ML
- Start with pretrained ArcFace.
- Keep ArcFace frozen for the first Model C implementation.
- Train the anti-impersonation branch in PyTorch.
- Use YAML/configuration for hyperparameters and thresholds.

## Adversarial training
- Face swap is the semantic manipulation.
- FGSM/PGD are model-aware image-space attacks.
- Keep attack generation separate from runtime inference.
- Implement FGSM before PGD.

## Runtime
- Avoid per-frame disk I/O.
- Inference failure must never become VERIFIED.
- Use temporal smoothing.
- Expose latency/FPS.

## Privacy and safety
- Use only consenting participants.
- Keep demo identity data local.
- Do not test against third-party authentication systems.

## Phase discipline
- Work only on the requested phase.
- Read current PROJECT_STATE before starting.
- At the end of every phase, run verification and produce a structured report.
- Do not start the next phase automatically.
