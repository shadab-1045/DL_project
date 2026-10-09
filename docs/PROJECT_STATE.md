# Project State

Update this file after every approved phase.

## Current phase
`Phase 1 — Foundation + Identity Engine`

## Approved architecture
- Model A: pretrained ArcFace identity recognizer.
- Model C: ArcFace identity representation + adversarially trained anti-impersonation branch.
- C-Control: internal ablation only.
- Deep-Live-Cam: live face-swap generator/test layer.
- SimSwap or equivalent: controlled offline swap generation.
- VGGFace2: identity data.
- FaceForensics++: manipulation/external evaluation data.
- FastAPI + WebSocket + React: application layer.

## Phase status
- Phase 0 — reconnaissance: `COMPLETED`
- Phase 1 — foundation + identity engine: `COMPLETED` (Pending ONNX Runtime CUDA resolution)
- Phase 2 — dataset + controlled swaps: `NOT_STARTED` (Blocked pending manual downloads)
- Phase 3 — anti-impersonation model + C-Control: `NOT_STARTED`
- Phase 4 — adversarial training: `NOT_STARTED`
- Phase 5 — full evaluation: `NOT_STARTED`
- Phase 6 — Deep-Live-Cam + app integration: `NOT_STARTED`
- Phase 7 — hardening + final demo: `NOT_STARTED`

## Checkpoints
- Model A: `buffalo_l` (InsightFace default)

## Dataset manifests
None.

## Experiment results
None.

## Known issues
- `onnxruntime-gpu` falls back to CPU because it requires system-level CUDA 12.x DLLs which are not currently in the Windows PATH, despite PyTorch detecting CUDA correctly.

## Update protocol
After every phase record:
- status
- files created/modified
- commands run
- tests
- metrics
- checkpoints
- deviations
- known issues
- next approved phase

Never mark a phase complete merely because code exists.
