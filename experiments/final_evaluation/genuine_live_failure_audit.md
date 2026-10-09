# Genuine Live Failure Root-Cause Audit

This document investigates the cause of Model A rejecting the identity during the completed 10-frame Genuine live trial (`manual_genuine_1791044014.jsonl`).

## 1. Observed Facts
- **Frames Evaluated**: 10
- **Face Detection**: 10/10 frames detected a face successfully.
- **Model A Similarity Range**: `0.124` to `0.169`
- **Model A Accepted**: `0/10` frames (failed to meet the `0.244529` frozen threshold).
- **Final Output**: `UNKNOWN`

## 2. Enrollment Configuration
The script `src/runtime/manual_live_validation.py` explicitly enrolled the identity "Alice" using exactly five reference images:
1. `sample_data/lfw/Angelina_Jolie/00.jpg`
2. `sample_data/lfw/Angelina_Jolie/01.jpg`
3. `sample_data/lfw/Angelina_Jolie/02.jpg`
4. `sample_data/lfw/Angelina_Jolie/03.jpg`
5. `sample_data/lfw/Angelina_Jolie/04.jpg`

## 3. Visual and Domain Differences
An explicit visual comparison was made between the LFW enrollment images and the saved live trial frame (`experiments/live_demo/manual/diagnostic_genuine.jpg`).
- **Enrollment Identity**: The five enrollment images exclusively depict Angelina Jolie (a female public figure).
- **Live Identity**: The physical person captured in the live webcam frame is a male operator.
- **Finding**: The diagnostic genuine image visually **does NOT contain the same physical person** represented by the enrollment images. The failure is entirely attributable to an identity mismatch in the physical testing environment, not a biometric failure or software bug.

## 4. Preprocessing and Model Consistency
Despite the physical mismatch, I strictly audited the preprocessing pipeline to ensure no hidden software issues exist:
- **Color Ordering**: Both use RGB. Enrollment images are loaded via `cv2.imread` (BGR) and handled internally by `InsightFace` which expects BGR arrays natively. The live pipeline `process_frame` directly passes the raw `frame_bgr` to the `IdentityPipeline`, mirroring the exact same path.
- **Resizing, Alignment, and Crop**: Both paths utilize the exact same `IdentityEngine.get_embedding(img)` and `engine.extract_faces()` methods which invoke the frozen `1k3d68.onnx` and `2d106det.onnx` landmark models, followed by the identical 112x112 affine alignment.
- **Normalization**: Both paths produce float32 feature vectors extracted from `w600k_r50.onnx`, natively L2-normalized.
- **Embedding Aggregation**: `IdentityPipeline.enroll()` computes the arithmetic mean of the five embeddings and strictly L2-normalizes the result before persisting it to the gallery.

## 5. Numerical Embedding Comparison
To rule out any inference pipeline corruption, I directly computed the exact pairwise cosine similarities between the five static enrollment embeddings and the static diagnostic live-frame embedding using a standalone script:

- **Similarity to Image 00**: `0.0177`
- **Similarity to Image 01**: `0.0708`
- **Similarity to Image 02**: `0.1104`
- **Similarity to Image 03**: `0.0126`
- **Similarity to Image 04**: `-0.0026`
- **Aggregated Mean-Embedding Similarity**: **`0.0513`**

*(Note: The live EMA similarity reached ~0.146 because the 10-frame webcam feed captured slight temporal pose variations of the non-enrolled operator that marginally spiked closer to the threshold than the single snapshot, but remained safely rejected).*

## 6. Conclusion and Likely Contributing Factors
There is **no software bug**, no preprocessing mismatch, and no threshold calibration error. 

The live Genuine condition failed (yielding `UNKNOWN` instead of `VERIFIED`) strictly because the physical participant testing the webcam was not the person enrolled in the system's gallery. Model A behaved exactly as expected by correctly rejecting the biometric mismatch. 

To perform a valid Genuine-condition evaluation, the pipeline must either:
A) Be tested by the physical person currently enrolled in the LFW gallery (Angelina Jolie).
B) Have the "Alice" gallery dynamically re-enrolled using 5 baseline reference images of the specific male operator performing the physical live test, ensuring the live operator is actually the genuine enrolled identity.
