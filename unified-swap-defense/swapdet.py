"""Inference-side swap detector (adversarially trained MobileNetV2). Input: BGR frame + ArcFace face bbox."""
import os
import numpy as np, cv2, torch, torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(ROOT, "models", "swapdet_deployed.pt")

class Net(nn.Module):  # takes [0,1] RGB so L-inf eps is in true pixel units
    def __init__(s, pretrained=False):
        super().__init__(); m = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if pretrained else None); m.classifier = nn.Sequential(nn.Dropout(.2), nn.Linear(1280, 1)); s.m = m
        s.register_buffer("mu", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)); s.register_buffer("sd", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
    def forward(s, x): return s.m((x - s.mu) / s.sd).squeeze(1)

def crop(img, bbox, m=0.35, out=224):  # must match gen_swap_data.crop (training-time crop)
    x1, y1, x2, y2 = bbox; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; s = max(x2 - x1, y2 - y1) * (1 + m)
    x1, y1 = int(cx - s / 2), int(cy - s / 2); s = int(s)
    big = cv2.copyMakeBorder(img, s, s, s, s, cv2.BORDER_REPLICATE)
    return cv2.resize(big[y1 + s:y1 + 2 * s, x1 + s:x1 + 2 * s], (out, out), interpolation=cv2.INTER_AREA)

class SwapDetector:
    def __init__(self, weights=WEIGHTS):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = Net().to(self.dev).eval(); self.net.load_state_dict(torch.load(weights, map_location=self.dev, weights_only=True))

    @torch.no_grad()
    def p_swap(self, frame_bgr, bbox):
        c = cv2.cvtColor(crop(frame_bgr, bbox), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(c).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.dev)
        return float(torch.sigmoid(self.net(x))[0])


def decide(p_swap, id_ok, identity, threshold=0.5, n_frames=None, min_frames=3):
    """Final verdict. The swap detector can veto an identity match (ArcFace alone accepts a good swap), and a positive
    verification needs at least `min_frames` of evidence (blocking is immediate)."""
    if p_swap is None:
        return "NO_FACE", "No face found in frame."
    if p_swap >= threshold:
        extra = f" ArcFace alone would have accepted '{identity}'." if id_ok else ""
        return "ATTACK_BLOCKED", f"Synthetic face detected (P(swap)={p_swap:.2f}).{extra}"
    if n_frames is not None and n_frames < min_frames:
        return "ANALYZING", f"Collecting evidence ({n_frames}/{min_frames} frames) before any verification."
    if not id_ok:
        return "UNKNOWN_IDENTITY", "Face looks real but matches no enrolled identity."
    return "SECURE_VERIFIED", f"Face looks real (P(swap)={p_swap:.2f}) and matches '{identity}'."
