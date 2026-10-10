# Documentation

> [!NOTE]
> **Layout.** This folder lives inside the `DL_project` repository. Wherever these pages say `DL_project/`, read *the repository root*. Run commands from this folder (`unified-swap-defense/`).

This project asks one question: **can a live face-swap fool a face-verification system, and what does it take to stop it?**

It is a university deep-learning project. The documentation describes what was built, what was measured, and where it falls short. Numbers come from the files in `eval_results/` and from scripts in this repository.

## Contents

| Document | What it covers |
|---|---|
| [Getting started](getting-started.md) | Install, enroll a user, run the API and UI, run the tests |
| [Architecture](architecture.md) | Components, request flow, verdict logic, repository layout |
| [Evaluation](evaluation.md) | Datasets, metrics, attack results, how to reproduce them |
| [Security and observability](security.md) | Request hardening in the API, telemetry, what is *not* protected |
| [Limitations](limitations.md) | What the system does not do, and open problems |

## The result in one paragraph

ArcFace alone accepts 99.6% of InSwapper face-swaps as the impersonated identity. Adding a MobileNetV2 swap detector that can veto an identity match cuts that to about 4% on a held-out test set, with 6.1% of real faces flagged per frame. That holds only against swaps that were not crafted against the detector. A white-box attacker with an L∞ budget of 1/255 drives detection to 0%, and the adversarially trained variants we tried were not usable (62–67% of real faces blocked). Details and sources are in [Evaluation](evaluation.md).

## Status of the pieces

| Piece | Status |
|---|---|
| Swap detector (`swapdet.py`, `models/swapdet_deployed.pt`) | In the live verdict path. Trained on one swapper (InSwapper). |
| ArcFace identity matching (`DL_project/src/`) | In the live verdict path. |
| Verdict logic (`swapdet.decide`) | In the live verdict path. Covered by `test_decide.py`. |
| FaceGuard Keras model (`FaceGuard-Digital-Forensic-System/`) | Bundled earlier project. Serves the legacy image/video/webcam tabs. **Not part of the swap verdict.** |
| `DL_project/experiments/` and `DL_project/docs/` | Historical research artifacts from earlier phases. Not re-verified here. |
