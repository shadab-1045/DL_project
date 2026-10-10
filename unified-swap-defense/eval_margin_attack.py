"""Gradient-masking check: re-attack every saved detector with a stronger, different attack than the BCE-PGD used in training/eval.
Attack: PGD-50 on the raw logit margin (no sigmoid/BCE, so saturation can't zero the gradient), 2 random restarts, worst case per sample.
A model whose BCE-PGD numbers were an artifact will collapse to ~0% here. Usage: python eval_margin_attack.py"""
import os, sys, glob, json
sys.argv = sys.argv[:1]
import torch
from swapdet import Net
from train_swapdet import DS, MODELS, OUT, dev

STEPS, RESTARTS, THR, BS = 50, 2, 0.5, 32
ds = DS("test", False); X = torch.stack([ds[i][0] for i in range(len(ds))]); Y = torch.tensor([float(c) for _, c in ds.items])
fk = torch.where(Y == 1)[0]

def margin_attack(model, x, eps):  # lower the logit of swapped faces as far as possible; keep the best (lowest-logit) restart
    best = torch.full((len(x),), 1e9, device=dev); best_adv = x.clone(); a = 2.5 * eps / STEPS
    for _ in range(RESTARTS):
        adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
        for _ in range(STEPS):
            adv.requires_grad_(True); g, = torch.autograd.grad(model(adv).sum(), adv)
            adv = torch.min(torch.max(adv.detach() - a * g.sign(), x - eps), x + eps).clamp(0, 1)
        with torch.no_grad(): lg = model(adv)
        m = lg < best; best[m] = lg[m]; best_adv[m] = adv[m]
    return best_adv

res = {}
ORDER = ["clean", "clean_finetune_control", "adversarial_eps1", "adversarial", "adversarial_eps1_w0.25"]   # deployment candidates first
for name in ORDER:
    f = os.path.join(MODELS, f"swapdet_{name}.pt")
    model = Net().to(dev).eval(); model.load_state_dict(torch.load(f, map_location=dev, weights_only=True))
    for p in model.parameters(): p.requires_grad_(False)
    r = {}
    for e in (1, 2, 4):
        det = []
        for i in range(0, len(fk), BS):
            x = X[fk[i:i + BS]].to(dev); xa = margin_attack(model, x, e / 255)
            with torch.no_grad(): det.append((torch.sigmoid(model(xa)) >= THR).float().cpu())
        r[f"swap_detected_under_margin_PGD50x2_eps{e}/255"] = float(torch.cat(det).mean())
    res[name] = r; print(name, json.dumps(r), flush=True)
    json.dump(res, open(os.path.join(OUT, "margin_attack_results.json"), "w"), indent=2)
