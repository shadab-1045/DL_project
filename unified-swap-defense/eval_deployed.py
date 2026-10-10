"""Operating points of the DEPLOYED detector (models/swapdet_deployed.pt) on the held-out test split, no adversary.
Attack success = ArcFace accepts the swap as the source identity AND the detector does not flag it (as in eval_system.py).
Writes eval_results/deployed_operating_point.json. Usage: python eval_deployed.py"""
import os, csv, json
import numpy as np, torch
from swapdet import Net
from train_swapdet import DS, DATA, OUT, MODELS

dev = "cuda" if torch.cuda.is_available() else "cpu"
ARC_THR = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "final_evaluation", "model_a_threshold.json")))["selected_threshold"]
meta = {r["tag"]: r for r in csv.DictReader(open(os.path.join(DATA, "meta.csv"))) if r["split"] == "test"}
ds = DS("test", False)
tags = [os.path.splitext(os.path.basename(f))[0] for f, _ in ds.items]
is_fake = np.array([c for _, c in ds.items])

m = Net().to(dev).eval(); m.load_state_dict(torch.load(os.path.join(MODELS, "swapdet_deployed.pt"), map_location=dev, weights_only=True))
p = []
with torch.no_grad():
    for i in range(0, len(ds), 32):
        p.append(torch.sigmoid(m(torch.stack([ds[j][0] for j in range(i, min(i + 32, len(ds)))]).to(dev))).cpu())
p = torch.cat(p).numpy()

paired = np.array([i for i in np.where(is_fake == 1)[0] if tags[i] in meta])   # swaps with a recorded ArcFace similarity
arc_ok = np.array([float(meta[tags[i]]["sim_to_source"]) >= ARC_THR for i in paired])
res = {"weights": "swapdet_deployed.pt", "n_fake": int((is_fake == 1).sum()), "n_real": int((is_fake == 0).sum()), "n_pairs": int(len(paired)),
       "arcface_threshold": ARC_THR, "attack_success_vs_ArcFace_only": float(arc_ok.mean()), "thresholds": {}}
for t in (0.3, 0.5, 0.8):
    res["thresholds"][str(t)] = {"swaps_caught": float((p[is_fake == 1] >= t).mean()), "real_flagged": float((p[is_fake == 0] >= t).mean()),
                                 "attack_success_vs_system": float((arc_ok & (p[paired] < t)).mean())}
json.dump(res, open(os.path.join(OUT, "deployed_operating_point.json"), "w"), indent=2); print(json.dumps(res, indent=1))
