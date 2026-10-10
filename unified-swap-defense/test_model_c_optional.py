"""Model C can be skipped in the live pipeline without ever producing VERIFIED. Stubs the heavy imports, so it needs only torch/numpy.
Usage: python test_model_c_optional.py"""
import os, sys, types
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root (holds src/)
for name, attr in (("src.runtime.identity_pipeline", "IdentityPipeline"), ("src.runtime.live_swap", "NativeInSwapper"),
                   ("src.models.model_c_v4", "ModelCV4"), ("src.preprocessing.alignment", "align_face")):
    mod = types.ModuleType(name); setattr(mod, attr, object); sys.modules[name] = mod
from src.runtime import live_pipeline_v4 as lp


class Face: bbox = np.array([10., 10., 110., 130.]); kps = None; normed_embedding = np.zeros(512)


class Engine:
    def __init__(self, faces): self.faces = faces
    def extract_faces(self, img): return self.faces


class Gallery:
    def identify(self, engine, emb, threshold): return {"similarity": 0.9, "identity": "Alice"}


class IdPipe:
    match_threshold = 0.2445
    def __init__(self, faces): self.engine, self.gallery = Engine(faces), Gallery()


def pipeline(faces):
    p = object.__new__(lp.LiveInferencePipelineV4)
    p.device, p.ema_alpha, p.identity_pipeline, p.swapper, p.c_adv_v4 = "cpu", 0.3, IdPipe(faces), None, None
    p.session_id = p.log_file = None; p.reset_temporal_state()
    return p


frame = np.zeros((200, 200, 3), np.uint8)
p = pipeline([Face()]); r = p.process_frame(frame, "genuine", 0)
assert r["model_a_raw_accept"] is True and r["matched_identity"] == "Alice"      # ArcFace result is still reported
assert r["final_state"] == "UNKNOWN" and r["raw_predicted_state"] == "UNKNOWN"    # no Model C => never VERIFIED
assert r["raw_c_class"] == -1
assert p.last_face_bbox is not None and p.last_processed_frame is frame          # hooks the unified API reads
p = pipeline([]); r = p.process_frame(frame, "genuine", 1)
assert r["face_detected"] is False and p.last_face_bbox is None                  # no face => bbox cleared

os.environ.pop("DL_LOAD_MODEL_C", None); assert lp._model_c_enabled() is True    # default: unchanged behaviour
os.environ["DL_LOAD_MODEL_C"] = "0"; assert lp._model_c_enabled() is False
print("model C optional: all cases pass")
