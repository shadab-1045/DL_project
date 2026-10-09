import json, numpy as np

fpath = "experiments/live_demo/manual/manual_genuine_1791044014.jsonl"
sims = []
with open(fpath, "r") as f:
    for line in f:
        d = json.loads(line)
        if "raw_model_a_similarity" in d: sims.append(d["raw_model_a_similarity"])
if sims: print(f"Sim: min={min(sims):.3f} max={max(sims):.3f} mean={np.mean(sims):.3f}")
