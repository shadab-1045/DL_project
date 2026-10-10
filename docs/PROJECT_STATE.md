# Project State

Update this file after every approved phase.

## Current phase
`Phase 7 — Hardening + final demo` (in progress)

## Approved architecture
- Model A: pretrained ArcFace identity recognizer.
- Model C: ArcFace identity representation + anti-impersonation branch (C-Control / C-Adv, later V4 fusion).
- C-Control: internal ablation only.
- Deep-Live-Cam / InSwapper: live face-swap generator/test layer.
- InSwapper: controlled offline swap generation.
- VGGFace2 / LFW: identity data.
- FaceForensics++: manipulation/external evaluation data.
- FastAPI + WebSocket + React: application layer.

## Phase status
Reports and artifacts exist for each phase below; they have not all been independently re-verified. See the correction notes
at the top of `experiments/model_c_adv/PHASE_4_SUMMARY.md`, `experiments/model_c_adv/robustness_summary.md` and `docs/PHASE_5_EVALUATION.md`.
- Phase 0 — reconnaissance: `COMPLETED`
- Phase 1 — foundation + identity engine: `COMPLETED`
- Phase 2 — dataset + controlled swaps: reports in `docs/dataset_build_report.md`, `experiments/data_qa/`
- Phase 3 — anti-impersonation model + C-Control: reports in `experiments/model_c_control/`
- Phase 4 — adversarial training: reports in `experiments/model_c_adv/` (FGSM only; accuracy under attack is low)
- Phase 5 — full evaluation: `docs/PHASE_5_EVALUATION.md`
- Phase 6 — Deep-Live-Cam + app integration: `experiments/final_evaluation/phase6*`
- Phase 7 — hardening + final demo: `experiments/live_demo/v4/`. Physical-webcam validation (7F.23) is `BLOCKED` and has not been performed.

## Checkpoints
- Model A: `buffalo_l` (InsightFace default)

## Dataset manifests
See `docs/DATA_AND_DATASET.md` and `docs/dataset_build_report.md`.

## Experiment results
See `experiments/` and `docs/PHASE_5_EVALUATION.md`.

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
