# Product Requirements Document

## 1. Product
**Adversarially Robust Face Identity Verification Against Synthetic Face-Swap Impersonation**

## 2. Problem
Ordinary identity recognition can identify a synthetic presentation as the enrolled identity. The system must therefore combine identity recognition with identity-aware anti-impersonation analysis.

## 3. Core scenarios

| Scenario | Identity | Authenticity | Final state |
|---|---|---|---|
| Genuine enrolled user | matched | genuine | `VERIFIED` |
| Different/unregistered person | no valid match | n/a | `UNKNOWN` |
| Face-swapped impersonation | may match | synthetic | `SUSPECTED_IMPERSONATION` |

## 4. Goals
- Provide reliable identity enrollment and recognition.
- Detect suspected synthetic impersonation of enrolled identities.
- Compare normal and adversarial anti-impersonation training.
- Evaluate robustness on held-out and cross-generator data.
- Demonstrate the final pipeline live with Deep-Live-Cam.
- Produce reproducible metrics and artifacts.

## 5. Non-goals
- Generic REAL-vs-FAKE classification as the main research task.
- Training a new face detector.
- Training ArcFace from scratch.
- Production deployment for real access control.
- Testing against systems belonging to third parties.

## 6. Functional requirements
- FR-01: enroll a user with several reference images.
- FR-02: generate/store identity embeddings.
- FR-03: perform identity matching.
- FR-04: support `UNKNOWN`.
- FR-05: run identity-aware anti-impersonation analysis.
- FR-06: use temporal smoothing for live decisions.
- FR-07: show identity + similarity + impersonation evidence + final state.
- FR-08: record experiment metadata and results.
- FR-09: support real-time inference.
- FR-10: provide an offline fallback demo path.

## 7. Technical requirements
- Identity: InsightFace/ArcFace.
- Anti-impersonation: PyTorch EfficientNet-B0 or Xception-based branch.
- Data processing: OpenCV/Python.
- Backend: FastAPI.
- Live transport: WebSocket.
- Frontend: React.
- Experiment tracking: TensorBoard plus JSON/CSV artifacts.

## 8. Success criteria
The project must:
- recognize genuine enrolled users;
- reject unrelated identities as unknown under calibrated thresholds;
- identify controlled face-swap impersonation cases with the anti-impersonation component;
- show measured difference between C-Control and C-Adv;
- evaluate unseen conditions/generators where practical;
- operate with live Deep-Live-Cam output;
- preserve reproducibility.

Do not define target metric values in advance. Measure them.

## 9. Privacy
Use public datasets according to their licenses and access requirements. Use only consenting live participants. Keep live face data local unless explicitly authorized.
