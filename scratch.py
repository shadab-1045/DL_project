import json, glob
import numpy as np

files = [
    "experiments/live_demo/manual/manual_genuine_1791043256.jsonl",
    "experiments/live_demo/manual/manual_different_person_1791043290.jsonl",
    "experiments/live_demo/manual/manual_impersonation_1791043312.jsonl"
]

for fpath in files:
    frames = 0
    faces = 0
    sims = []
    accepts = 0
    c_advs = 0
    c_probs = []
    c_classes = []
    states = []
    lats = []
    with open(fpath, "r") as f:
        for line in f:
            d = json.loads(line)
            frames += 1
            if d.get("face_detected"): faces += 1
            if "raw_model_a_similarity" in d: sims.append(d["raw_model_a_similarity"])
            if d.get("model_a_raw_accept"): accepts += 1
            if "raw_c_probs" in d and d.get("model_a_raw_accept"): 
                c_advs += 1
                c_probs.append(d["raw_c_probs"])
                c_classes.append(d["raw_c_class"])
            states.append(d.get("final_state"))
            if "total_latency" in d: lats.append(d["total_latency"])
    
    cond = fpath.split("_")[-2] if "person" not in fpath else "different_person"
    print(f"\nCondition: {cond.upper()}")
    print(f"Frames: {frames}, Faces: {faces}")
    if sims: print(f"Sim: min={min(sims):.3f} max={max(sims):.3f} mean={np.mean(sims):.3f}")
    print(f"Accepts: {accepts}, C-Adv: {c_advs}")
    if c_advs > 0:
        p = np.mean(c_probs, axis=0)
        print(f"C-Adv Probs: G={p[0]:.3f} D={p[1]:.3f} I={p[2]:.3f}")
        for c in set(c_classes): print(f"Class {c}: {c_classes.count(c)}")
    for s in set(states): print(f"State {s}: {states.count(s)}")
    if lats: print(f"Lat: min={min(lats):.3f} max={max(lats):.3f} mean={np.mean(lats):.3f}")
