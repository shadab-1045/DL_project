# Unified swap defense

A face-swap attack harness and swap detector built on top of this repository's identity pipeline (ArcFace + InSwapper). It answers one question: **can a live face-swap fool identity verification, and what does it take to stop it?**

| Question | Answer, measured on a held-out test set |
|---|---|
| Does ArcFace verification stop an InSwapper swap? | **No.** It accepts **99.6%** of swaps as the impersonated person (n = 262). |
| Does adding a swap detector help? | **Yes**, against non-adaptive swaps: **96.2%** caught, **6.1%** of real frames flagged. |
| Does it survive an attacker who targets the detector? | **No.** A 1/255 perturbation drops detection to **0%**. |

Full write-up, including the experiments that failed: [`docs/`](docs/index.md).

## How this relates to the rest of the repository

This folder **adds to** the research code in the repository root and reuses its identity pipeline. It does not replace it.

| Part | Where | Who |
|---|---|---|
| Identity engine, enrollment, live pipeline, Model C research (`src/`, `experiments/`, root `docs/`) | repository root | Shadab |
| Swap detector, verdict logic, unified API, evaluation scripts, `docs/` | this folder | Arsal |
| FaceGuard (Keras deepfake classifier + React UI) | `FaceGuard-Digital-Forensic-System/` | Bundled earlier project, with its own README and licence |

> This work is also maintained as a standalone repository: [NotArsal/DL-Adverserial](https://github.com/NotArsal/DL-Adverserial).
>
> In the standalone version the repository root was called `DL_project/`. Here, wherever the docs say `DL_project/`, read **the repository root**. This folder's scripts have been adapted to that layout.

## What it needs from the root repository

- `src/runtime/live_pipeline_v4.py` (`LiveInferencePipelineV4`), which the API loads. It requires the Model C V4 checkpoint at `experiments/model_c_v4/best_fusion_model.pt` and `models/inswapper_128.onnx`. Neither is stored in git.
- The two-line hook in `src/runtime/live_pipeline_v4.py` and the path sanitisation in `src/api/server.py` added by this change.

Running the verdict endpoint therefore also runs Model C V4 on every frame whose identity matched, although the verdict ignores its output. This is a known cost; see the PR description.

## Run

Run everything **from this folder**.

```bash
pip install -r requirements.txt
pip install "../insightface-0.7.3-cp310-cp310-win_amd64.whl"   # Windows only

python make_demo_identity.py                # exports the "victim" identity the attack impersonates
python enroll_user.py YourName --webcam     # enroll yourself (6 snapshots)
python test_decide.py                       # verdict-logic checks, no models needed beyond torch

uvicorn unified_api:app --host 127.0.0.1 --port 8001

# second terminal
cd FaceGuard-Digital-Forensic-System/frontend
npm install && npm start                    # http://localhost:3005
```

Open the **Unified Checkpoint** tab. More detail: [Getting started](docs/getting-started.md).

## Contents

| Path | Purpose |
|---|---|
| `swapdet.py` | Swap detector and `decide()` verdict logic |
| `unified_api.py` | API: `/ws/unified-verify` plus mounts for the identity pipeline and FaceGuard |
| `gen_swap_data.py`, `train_swapdet.py` | Build the swap dataset, train clean / control / adversarial detectors |
| `eval_system.py`, `eval_margin_attack.py`, `eval_randomized.py` | End-to-end and white-box evaluation |
| `eval_faceguard.py`, `eval_adversarial.py` | FaceGuard classifier evaluation |
| `models/swapdet_deployed.pt` | Deployed detector (9 MB) |
| `eval_results/` | Metrics and figures |
| `docs/` | Architecture, evaluation, security, limitations |

## CI

`.github/workflows/ci.yml` runs on changes to this folder:

| Job | Gates |
|---|---|
| `python` | syntax / undefined names (ruff), compile, `test_decide.py` |
| `frontend` | FaceGuard UI `npm ci --legacy-peer-deps` and production build (compile errors fail; lint warnings are reported); `npm audit` is reported but non-blocking |

It does **not** run the models, the API or the evaluation scripts: those need GPU weights and datasets that are not in git.
Model quality is therefore not gated by CI; the numbers in `eval_results/` are from local runs.
