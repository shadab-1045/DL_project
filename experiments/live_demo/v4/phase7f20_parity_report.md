# Phase 7F.20: Correct Live JPEG Preprocessing Parity

## 1. Dataset Preprocessing Verification
An audit of `src/data/generate_controlled_swaps_v2.py` established the exact pre-processing sequence that the V4 model expects:
1. `align_face(image_bgr, kps)` produces a `112x112` RGB array.
2. The dataset generator executed `cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)`.
3. The image was saved to disk via `cv2.imwrite` using OpenCV's default parameters (which encodes as JPEG with a default quality of 95).
4. The PyTorch `Dataset` class later loaded this file using `Image.open(img_path).convert('RGB')`.
5. The PIL Image was then resized to 224x224 and normalized.

Because the live inference pipeline (`LiveInferencePipelineV4`) originally bypassed the `cv2.imwrite` step and piped the array directly to memory, it effectively presented the EfficientNet visual branch with an out-of-distribution (uncompressed) image, triggering False Positives on Genuine faces.

## 2. The Isolated Parity Adapter
An isolated parity adapter (`src/preprocessing/v4_live_parity.py`) was implemented. 
Rather than writing to disk for every live webcam frame (which would introduce extreme IO latency), the adapter mathematically reproduces the exact dataset JPEG compression distribution strictly in-memory using OpenCV's standard encoding bindings:

```python
# 1. Simulate dataset save: cv2.imwrite converts to BGR and saves as .jpg
aligned_bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)
success, encoded = cv2.imencode('.jpg', aligned_bgr)
            
# 2. Simulate dataset load: Image.open(path).convert('RGB')
image_pil = Image.open(io.BytesIO(encoded.tobytes())).convert('RGB')
```

This adapter was placed strictly within the V4 inference branch of the `LiveInferencePipelineV4`. It does not affect Model A's embeddings or the UI display frame.

## 3. Parity Test Results
A rigorous offline parity test (`scratch/test_v4_live_parity.py`) verified the implementation mathematically against the physical disk path:

- **Shape & Dtype**: Match exactly (`[1, 3, 224, 224]`, `torch.float32`)
- **Numerical Parity**: The maximum absolute pixel difference between the physical disk output tensor and the in-memory adapter tensor is exactly `0.000000`. The adapter perfectly simulates the dataset path.
- **Visual Branch Logit**: Both the adapter and the disk path emitted an identical logit of `10.1494` for the test image, whereas the old bypass path emitted `8.7539`.
- **Identity Branch**: Outputs were mathematically unaffected (`P(Id_Match) = 1.0000` for both).
- **Frozen Integrity**: SHA-256 hashes for the V4 fusion checkpoint remained perfectly identical before and after.

## 4. Performance & Integration
The overhead latency of the in-memory parity adapter was measured at exactly **0.002 seconds (2 milliseconds)** on CPU, rendering it effectively cost-free for real-time video processing.

The original `LiveInferencePipelineV4` has been updated to use the adapter. The V4 offline quantitative evaluations remain completely untouched. 

*Conclusion*: The V4 live pipeline now perfectly matches its dataset distribution. The preprocessing discrepancy has been robustly addressed.
