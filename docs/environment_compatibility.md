# Environment and Dependency Compatibility

## Current Installed Versions
- **Python**: 3.10.11
- **PyTorch**: 2.5.1+cu121 (CUDA is functional via internal binaries)
- **ONNX Runtime GPU**: 1.23.2
- **InsightFace**: 2.0
- **OpenCV**: 5.0.0.93

## Identified Compatibility Issue
PyTorch successfully detects the CUDA 12.1 environment (via `torch.cuda.is_available()`) because PyTorch wheels bundle their own CUDA runtime binaries (cuDNN, cuBLAS). 

However, `onnxruntime-gpu` attempts to locate system-level CUDA DLLs in the Windows PATH (`cublas64_12.dll`, `cudnn64_*.dll`). Since the host system does not have the NVIDIA CUDA Toolkit 12.x installed globally on the PATH, ONNX Runtime fails to initialize the `CUDAExecutionProvider` and falls back to `CPUExecutionProvider`. 

**Impact:** InsightFace runs face detection and feature extraction on the CPU, causing slower inference times (100-300ms per frame) during identity matching. 

## Strategy and Recommendation
Do **NOT** downgrade InsightFace, PyTorch, or randomly copy DLLs in this environment, as it could destabilize PyTorch's native CUDA hooks required for Model C training.

**Recommended Approach:**
Since Model C (the anti-impersonation branch) will be trained entirely in PyTorch (which correctly utilizes the RTX 3050), the current CPU fallback for ArcFace is acceptable for Phase 1 and data generation. 
For Phase 6 (Live integration with Deep-Live-Cam), where FPS is critical:
- **Unified Environment:** Install the NVIDIA CUDA Toolkit 12.x and cuDNN system-wide and add them to the Windows PATH so that `onnxruntime-gpu` and Deep-Live-Cam can share the GPU natively.
- **Separate Process:** If dependency constraints between Deep-Live-Cam and Model C become intractable, run Deep-Live-Cam in a separate conda/venv with its own dependencies, and stream frames in memory or via localhost to the PyTorch pipeline.
