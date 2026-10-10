# Getting started

> [!NOTE]
> **Layout.** This folder lives inside the `DL_project` repository. Wherever these pages say `DL_project/`, read *the repository root*. Run commands from this folder (`unified-swap-defense/`).

## Requirements

| Need | Notes |
|---|---|
| Python 3.10 | The bundled InsightFace wheel is `cp310` / Windows. |
| Node.js 18+ | For the React UI. |
| NVIDIA GPU (recommended) | Developed on an RTX 4060 8 GB. PyTorch and ONNX Runtime use CUDA when available and fall back to CPU. |
| A webcam | Needed for `enroll_user.py --webcam` and the live checkpoint. |
| Model files not in git | `*.onnx`, `*.h5`, `*.pt` are git-ignored, except `models/swapdet_deployed.pt`. You need the InSwapper weights (`DL_project/models/inswapper_128.onnx`) and the FaceGuard weights (`FaceGuard-Digital-Forensic-System/models/faceguard_phase2_finetuned.h5`) in place. |

TensorFlow runs on CPU only in this project (`unified_api.py` hides the GPU from it) so it does not compete with PyTorch for VRAM.

## Run locally

```bash
# 1. Install
pip install -r requirements.txt
pip install "../insightface-0.7.3-cp310-cp310-win_amd64.whl"   # Windows only (wheel sits at the repository root)

# 2. Create the identities
python make_demo_identity.py              # exports photos of the "victim" the attack impersonates (default: Angelina_Jolie from LFW)
python enroll_user.py YourName --webcam   # enrolls you from 6 webcam snapshots

# 3. API
uvicorn unified_api:app --host 127.0.0.1 --port 8001

# 4. UI (second terminal)
cd FaceGuard-Digital-Forensic-System/frontend
npm install
npm start                                  # http://localhost:3005
```

Open the **Unified Checkpoint** tab. Choose *genuine* to verify yourself, or *impersonation* to have the server apply a live InSwapper swap to your frames and watch the detector respond.

Enroll in the same camera and lighting you will use. The detector threshold was chosen with one user's webcam frames in view (see [Limitations](limitations.md)).

If the camera feed is black, `enroll_user.py` says so and suggests another index: `python enroll_user.py YourName --webcam 1`.

## Tests and checks

| Command | What it checks | Needs |
|---|---|---|
| `python test_decide.py` | The verdict logic: all five verdicts, fail-closed boundary, evidence minimum. | Nothing heavy |
| `python smoke_unified.py` | Boots the full API with real models and sends frames over the WebSocket. | Models, GPU or patience |
| `python bench_live.py` | Per-frame latency of the live path, and which execution providers are active. | Models |

`bench_live.py` prints the numbers for *your* machine. This documentation does not quote a latency figure, because none was re-measured when it was written.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| "Attack simulation needs a source identity" | `make_demo_identity.py` not run, or server not restarted after it. |
| ONNX Runtime silently using CPU | Wrong CUDA/cuDNN build. `requirements.txt` pins `onnxruntime-gpu==1.20.0` for CUDA 12 + cuDNN 9. `bench_live.py` prints the active providers. |
| Out-of-memory or Windows paging-file errors | Running swap-generation workers and GPU training together. Run them one at a time; use batch 32 for training on 8 GB. |
| Browser blocked by the API | Your UI origin is not in `ALLOWED_ORIGINS` (default: `localhost`/`127.0.0.1` on ports 3000 and 3005). |
