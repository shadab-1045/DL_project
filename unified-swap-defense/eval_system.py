"""End-to-end attack-vs-defense numbers on the held-out (identity-disjoint) test split.
Attack success = the swapped face is accepted as the source identity by the *whole* system.
 - ArcFace only      : accepted if cos-sim(swap, source) >= ArcFace threshold (calibrated on validation by the DL_project pipeline)
 - + swap detector   : additionally requires P(swap) < 0.5
 - adaptive attacker : white-box margin-PGD on the fake crop against the detector (ASSUMES the perturbation leaves ArcFace similarity
                       unchanged; ArcFace is not attacked here, so this is the detector's worst case, not a full-system proof).
Usage: python eval_system.py"""
import os, csv, json
import numpy as np, torch
from torch.utils.data import DataLoader
from swapdet import Net
from train_swapdet import DS, margin_pgd, DATA, OUT, MODELS, dev

ARC_THR = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "final_evaluation", "model_a_threshold.json")))["selected_threshold"]
meta = {r["tag"]: r for r in csv.DictReader(open(os.path.join(DATA, "meta.csv"))) if r["split"] == "test"}
ds = DS("test", False)
tags = [os.path.splitext(os.path.basename(f))[0] for f, _ in ds.items]
is_fake = np.array([c for _, c in ds.items]); fake_idx = np.array([i for i in np.where(is_fake == 1)[0] if tags[i] in meta])  # only swaps with recorded ArcFace sims
X = torch.stack([ds[i][0] for i in range(len(ds))])
arc_accept = np.array([float(meta[tags[i]]["sim_to_source"]) >= ARC_THR for i in fake_idx])
res = {"n_test_pairs": int(len(fake_idx)), "arcface_threshold": ARC_THR,
       "attack_success_vs_ArcFace_only": float(arc_accept.mean())}

def probs(model, idx, att=None):
    out = []
    for i in range(0, len(idx), 32):
        x = X[idx[i:i + 32]].to(dev); y = torch.ones(len(x), device=dev)
        if att: x = att(model, x, y)
        with torch.no_grad(): out.append(torch.sigmoid(model(x)).cpu())
    return torch.cat(out).numpy()

import glob
NAMES = [os.path.basename(f)[8:-3] for f in sorted(glob.glob(os.path.join(MODELS, "swapdet_*.pt"))) if "deployed" not in f]
for name in NAMES:
    m = Net().to(dev).eval(); m.load_state_dict(torch.load(os.path.join(MODELS, f"swapdet_{name}.pt"), map_location=dev, weights_only=True))
    p_fake, p_real = probs(m, fake_idx), probs(m, np.where(is_fake == 0)[0])
    r = {"genuine_user_blocked(false_alarm)": float((p_real >= .5).mean()), "swap_detected": float((p_fake >= .5).mean()),
         "attack_success_vs_system": float((arc_accept & (p_fake < .5)).mean())}
    for e in (1, 2, 4):  # attacker minimises the swap logit with strong margin-PGD (50 steps x 2 restarts)
        pa = probs(m, fake_idx, lambda mm, x, y: margin_pgd(mm, x, e / 255))
        r[f"attack_success_vs_system_margin_pgd_eps{e}/255"] = float((arc_accept & (pa < .5)).mean())
    r["operating_points(threshold: swap_detected / real_false_alarm)"] = {str(t): [float((p_fake >= t).mean()), float((p_real >= t).mean())] for t in (0.5, 0.7, 0.8, 0.9, 0.95)}
    res[name] = r; print(name, json.dumps(r, indent=1), flush=True)
json.dump(res, open(os.path.join(OUT, "system_results.json"), "w"), indent=2)
