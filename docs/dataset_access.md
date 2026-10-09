# Dataset Access and Acquisition

## VGGFace2
- **Role**: Primary identity-organized face dataset.
- **Access Requirements**: The dataset is not freely downloadable without registration. Researchers must typically fill out an access request or use an approved mirror.
- **Acquisition Procedure**:
  1. Obtain access from the official maintainers or use a documented mirror (e.g., Kaggle or academic torrents) in compliance with the license.
  2. Extract the archive into `data/raw/vggface2/`.
  3. The structure should be: `data/raw/vggface2/<identity_id>/<image_file>.jpg`.
- **Blocker Status**: Currently blocked in this automated environment due to missing credentials/approval. We are using the LFW subset for pipeline validation.

## FaceForensics++ (FF++)
- **Role**: FaceSwap manipulation data and external evaluation data.
- **Access Requirements**: Requires filling out a Google Form provided by the FaceForensics++ team to receive an automated email with the download script and credentials.
- **Acquisition Procedure**:
  1. Complete the FaceForensics++ Terms of Use form.
  2. Use the provided `download-FaceForensics.py` script.
  3. Download the `FaceSwap` manipulated videos and the `original_sequences`.
  4. Extract frames into `data/raw/faceforensics/`.
- **Blocker Status**: Blocked due to required manual form submission.

## SimSwap (Controlled Swap Generator)
- **Role**: Generates project-specific face-swaps.
- **Repository**: `https://github.com/neuralchen/SimSwap`
- **Pinned Commit**: `982ab91` (or latest stable compatible branch).
- **Blocker Status**: Requires downloading large pre-trained weights (e.g., `arcface_checkpoint.tar`, `checkpoints/people/`). For the Phase 2 pipeline validation, we simulate the generator's output if the model weights are not loaded.
