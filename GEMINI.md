# Adversarial Face Identity Project — Master Context

## Project
**Adversarially Robust Face Identity Verification Against Synthetic Face-Swap Impersonation**

This is an identity-aware anti-impersonation research system, not a generic REAL-vs-FAKE classifier.

## Core problem
A standard face-recognition system may identify a face-swapped presentation as an enrolled person's identity because the visible face has been synthetically changed.

The system therefore separates:
1. **Identity:** who does the face resemble?
2. **Presentation authenticity:** is that identity genuinely presented or synthetically presented?

## Deployed systems

### Model A — Baseline identity recognizer
- Pretrained InsightFace/ArcFace.
- Identity recognition only.
- No real/fake classifier training.
- No adversarial training.
- Enrollment is performed by extracting embeddings from several reference images.
- Output: identity + similarity.

### Model C — Proposed adversarial anti-impersonation system
- Reuses the same ArcFace identity representation.
- Adds a trainable visual anti-impersonation branch.
- Inputs: current face/context crop + ArcFace embedding of the best-matching claimed identity.
- Internal classes:
  - `genuine`
  - `different_person`
  - `impersonation`
- Training includes genuine examples, different-person negatives, controlled face-swap impersonation examples, hard manipulation variants, and classical adversarial examples (FGSM; PGD optional).
- Final output:
  - `VERIFIED`
  - `UNKNOWN`
  - `SUSPECTED_IMPERSONATION`

### Research control
**C-Control** uses the same anti-impersonation architecture as Model C but is trained without adversarial-example generation. It is an internal ablation, not a required final UI model. It isolates the benefit of adversarial training.

## External components
- **Deep-Live-Cam:** live face-swap generator / final real-time test layer.
- **SimSwap or equivalent:** offline controlled face-swap generation.
- **FaceForensics++ FaceSwap:** additional manipulated-video and external evaluation data.
- **VGGFace2:** identity-labeled data and controlled source/target identities.
- **InsightFace/ArcFace:** identity recognition.
- **PyTorch:** anti-impersonation training.
- **OpenCV:** frame/image processing.
- **FastAPI + WebSocket:** backend/live transport.
- **React:** frontend.

## Runtime flow
Webcam → Deep-Live-Cam → processed frame → face detection/alignment → ArcFace identity embedding → gallery match → Model C anti-impersonation inference → temporal smoothing → decision engine → FastAPI/WebSocket → React dashboard.

## Key live scenarios
1. Genuine enrolled user → high identity similarity + low impersonation evidence → `VERIFIED`.
2. Different/unknown person → low identity similarity → `UNKNOWN`.
3. Another person with enrolled identity face-swapped onto them → potentially high identity similarity, but high synthetic-presentation evidence → `SUSPECTED_IMPERSONATION`.

## Data model
A training instance is a relationship, not `image -> real/fake`:

`reference identity + probe sample -> label`

Important metadata:
- `reference_identity`
- `visible_identity`
- `physical_identity`
- `label`
- `attack_type`
- `generator`
- `source_dataset`
- `split`

## Non-negotiable rules
- Do not convert the project into a generic real/fake classifier.
- Do not train ArcFace from scratch for the main project.
- Keep ArcFace frozen for the first Model C implementation.
- Split identities/videos before generating pairs.
- Never allow frames from the same source video to cross train/validation/test.
- Do not train on final live-demo participants.
- Do not claim adversarial-training improvement without C-Control.
- Calibrate thresholds on validation data only.
- Record seeds, configs, checkpoints, manifests, and metrics.
- Use consenting participants for the live demo.
- Never fabricate experimental numbers.

## Primary research question
> Does adversarial training improve robustness of an identity-aware anti-impersonation model against synthetic face-swap presentations while preserving genuine-user verification performance?

## Development philosophy
Build in gated phases. Do not implement the whole project in one task. Stop after each phase and provide a structured verification report.
