# DL_project: adversarially robust face identity verification

Course project on face identity verification under face-swap impersonation. The repository holds the research pipeline (identity engine, anti-impersonation models, datasets, experiments, live demo) and a separate folder that tests how a face-swap detector holds up.

## Where things are

| Path | Contents |
|---|---|
| `src/` | Identity engine (ArcFace), anti-impersonation models, training, evaluation, runtime pipelines and the FastAPI server |
| `experiments/` | Training logs, metrics, reports and live-demo records for each phase |
| `docs/` | Project spec, architecture, dataset notes and phase reports. Start with [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md) |
| `frontend/` | React + Vite frontend for the live demo |
| `scripts/`, `tests/` | Dataset building scripts and tests |
| [`unified-swap-defense/`](unified-swap-defense/README.md) | Swap detector, verdict API, white-box evaluation and documentation, built on top of the identity pipeline |

## Reading the results

Several phase reports were written before later checks and now carry correction notes at the top. Read them before quoting any number:

- [`docs/PHASE_5_EVALUATION.md`](docs/PHASE_5_EVALUATION.md)
- [`experiments/model_c_adv/PHASE_4_SUMMARY.md`](experiments/model_c_adv/PHASE_4_SUMMARY.md)
- [`experiments/model_c_adv/robustness_summary.md`](experiments/model_c_adv/robustness_summary.md)

The swap-defense results, including the failed experiments, are in [`unified-swap-defense/docs/evaluation.md`](unified-swap-defense/docs/evaluation.md).

## Running

Model weights and datasets are not stored in git. See [`docs/environment_compatibility.md`](docs/environment_compatibility.md) for the environment and [`unified-swap-defense/docs/getting-started.md`](unified-swap-defense/docs/getting-started.md) for the swap-defense demo.

## Credits

| Part | Author |
|---|---|
| Research pipeline: `src/`, `experiments/`, `docs/`, `frontend/`, `scripts/`, `tests/` | [Shadab](https://github.com/shadab-1045) |
| `unified-swap-defense/`: swap detector, verdict logic, unified API, evaluation scripts and documentation | [Arsal](https://github.com/NotArsal), also maintained standalone at [NotArsal/DL-Adverserial](https://github.com/NotArsal/DL-Adverserial) |
| `unified-swap-defense/FaceGuard-Digital-Forensic-System/` | Bundled earlier project, with its own README and licence (Copyright Hiba V S) |
