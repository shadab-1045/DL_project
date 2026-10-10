import os
import numpy as np
import cv2

IMG_SIZE = (224, 224)
MODEL_PATH = os.getenv("MODEL_PATH", "../models/faceguard_phase2_finetuned.h5")

_model = None


def load_faceguard_model():
    """Load and cache the FaceGuard model."""
    global _model
    if _model is not None:
        return _model

    try:
        from tensorflow.keras.models import load_model
        _model = load_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
        return _model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")


def preprocess_frame(img_rgb: np.ndarray) -> np.ndarray:
    """
    Resize, normalize, and batch a single RGB frame for model inference.
    Input:  HxWx3 uint8 RGB numpy array
    Output: 1x224x224x3 float32 tensor
    """
    resized = cv2.resize(img_rgb, IMG_SIZE)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)
