# Dataset Build Report

## A. Phase 2B Checkpoint

The human operator has supplied the `inswapper_128.onnx` deep-learning swap model, which was validated via an automated smoke test (`scripts/inswapper_smoke_test.py`). 
The inconsistency regarding `insightface` versioning has been resolved by installing the correct `0.7.3` wheel provided by the human operator.

However, the actual research data ingestion remains **BLOCKED**.

### Required Manual Ingestion:
1. **VGGFace2 (Primary Genuine Dataset)**:
   - **Status**: Not found in `data/raw/vggface2`.
   - **Blocker**: The dataset requires manual authorization (Kaggle login/token or Google Drive access) to download the large 40GB archive.
   - **Action**: Please provide `kaggle.json` credentials or a direct authorized download link/method for the full VGGFace2 dataset.

2. **FaceForensics++ (Manipulation Dataset)**:
   - **Status**: Not found in `data/raw/faceforensics`.
   - **Blocker**: Requires completing the official Google Form.

### Pipeline Readiness
The data pipeline scripts (`build_identity_split.py`, `build_samples_manifest.py`, `generate_controlled_swaps.py`, `build_training_pairs.py`, `validate_dataset.py`) have been strictly refactored to point to `data/raw/vggface2` and require actual `inswapper_128.onnx` models. They will exit immediately without executing if the data is not present. They are fully prepared to run against the data once provided.
