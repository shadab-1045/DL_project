"""Summary figures from eval_results/*.json -> eval_results/summary.png"""
import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results")
sysr = json.load(open(os.path.join(R, "system_results.json"))); rnd = json.load(open(os.path.join(R, "randomized_defense_clean.json")))
mar = json.load(open(os.path.join(R, "margin_attack_results.json"))); sw = json.load(open(os.path.join(R, "swapdet_results.json")))
dep = json.load(open(os.path.join(R, "deployed_operating_point.json")))   # python eval_deployed.py
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

# (a) attack success against the verification system
labels = ["ArcFace only", "+ detector\nthr 0.3 (deployed)\nno adversary", "+ detector\nthr 0.5\nwhite-box ε=1/255"]
vals = [sysr["attack_success_vs_ArcFace_only"], dep["thresholds"]["0.3"]["attack_success_vs_system"], sysr["clean"]["attack_success_vs_system_margin_pgd_eps1/255"]]
b = ax[0].bar(labels, [v * 100 for v in vals], color=["#c0392b", "#27ae60", "#c0392b"])
for r, v in zip(b, vals): ax[0].text(r.get_x() + r.get_width() / 2, v * 100 + 1.5, f"{v * 100:.1f}%", ha="center")
ax[0].set(ylabel="swap attack success (%)", title="Swap attack success (n=262 held-out pairs)", ylim=(0, 110))

# (b) detection under white-box attack
eps = [0, 1, 2, 4]
ax[1].plot(eps, [94.7] + [mar["clean"][f"swap_detected_under_margin_PGD50x2_eps{e}/255"] * 100 for e in eps[1:]], "o-", label="clean detector (margin-PGD)")
ax[1].plot(eps, [87.8] + [rnd["randomized"][f"swap_detected_under_adaptive_marginEOT_PGD30_eps{e}/255"] * 100 for e in eps[1:]], "s-", label="randomized transforms (adaptive EOT)\n[false alarms 16.7%]")
ax[1].set(xlabel="L∞ budget ε (x/255)", ylabel="swaps detected (%)", title="Detection under white-box attack", ylim=(0, 100)); ax[1].legend(loc="upper right", fontsize=8); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(R, "summary.png"), dpi=140); print("saved summary.png")
