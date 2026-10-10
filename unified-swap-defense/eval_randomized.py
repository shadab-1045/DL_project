"""Randomized-input-transformation defense vs an ADAPTIVE attacker (EOT-PGD).
Defense: p(x) = mean_k sigmoid(f(T_k(x))), T = random {gaussian noise, blur, down/up-resize}. No retraining: uses the clean-trained detector
(which saw noise/blur/resize/JPEG augmentation). Attacker: margin-PGD-30 whose gradient averages K_ATT random draws per step (EOT), i.e. it knows the defense.
Reports detection of swaps (attack on fakes only) and false alarms on reals, plain vs randomized. Usage: python eval_randomized.py [MODEL_NAME]"""
import os, sys, json
import numpy as np, torch, torch.nn.functional as F
from swapdet import Net

NAME = sys.argv[1] if len(sys.argv) > 1 else "clean"
sys.argv = sys.argv[:1]   # train_swapdet parses argv at import
from train_swapdet import DS, MODELS, OUT, dev
K_DEF, K_ATT, STEPS, THR, BS = 8, 4, 30, 0.5, 32
model = Net().to(dev).eval(); model.load_state_dict(torch.load(os.path.join(MODELS, f"swapdet_{NAME}.pt"), map_location=dev, weights_only=True))
for p in model.parameters(): p.requires_grad_(False)

def gauss_kernel(sig):
    r = int(3 * sig) + 1; x = torch.arange(-r, r + 1, device=dev).float(); k = torch.exp(-x ** 2 / (2 * sig ** 2)); return k / k.sum()

def T(x):  # differentiable random transform; one fresh draw of (scale, blur, noise) per call, applied to the whole batch
    s = float(np.random.uniform(0.6, 1.0)); x = F.interpolate(F.interpolate(x, scale_factor=s, mode="bilinear", antialias=True), size=(224, 224), mode="bilinear")
    k = gauss_kernel(float(np.random.uniform(0.3, 1.2))); r = len(k) // 2
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(3, 1, 1, 1), groups=3)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(3, 1, 1, 1), groups=3)
    return (x + torch.randn_like(x) * float(np.random.uniform(0.01, 0.04))).clamp(0, 1)

def p_plain(x): return torch.sigmoid(model(x))
def p_rand(x, k=K_DEF): return torch.stack([torch.sigmoid(model(T(x))) for _ in range(k)]).mean(0)

def attack(x, eps, defense):  # minimise the (EOT-averaged) swap logit; EOT over K_ATT random draws per step when the defense is randomized
    x = x.detach(); adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1); a = 2.5 * eps / STEPS
    for _ in range(STEPS):
        adv.requires_grad_(True)
        lg = torch.stack([model(T(adv)) for _ in range(K_ATT)]).mean(0) if defense == "rand" else model(adv)
        g, = torch.autograd.grad(lg.sum(), adv)
        adv = torch.min(torch.max(adv.detach() - a * g.sign(), x - eps), x + eps).clamp(0, 1)
    return adv.detach()

ds = DS("test", False); X = torch.stack([ds[i][0] for i in range(len(ds))]); Y = torch.tensor([float(c) for _, c in ds.items])
fk, rl = torch.where(Y == 1)[0], torch.where(Y == 0)[0]
def run(idx, fn, att=None, eps=0):
    out = []
    for i in range(0, len(idx), BS):
        x = X[idx[i:i + BS]].to(dev)
        if att: x = attack(x, eps, att)
        with torch.no_grad(): out.append(fn(x).cpu())
    return torch.cat(out).numpy()

res = {"model": NAME, "K_defense": K_DEF, "K_attack_EOT": K_ATT, "pgd_steps": STEPS, "n_fake": len(fk), "n_real": len(rl)}
for dname, fn in (("plain", p_plain), ("randomized", p_rand)):
    r = {"clean_swap_detected": float((run(fk, fn) >= THR).mean()), "clean_real_false_alarm": float((run(rl, fn) >= THR).mean())}
    for e in (1, 2, 4):  # attacker perturbs the swapped image to look real; adaptive = knows defense (EOT)
        r[f"swap_detected_under_adaptive_marginEOT_PGD30_eps{e}/255"] = float((run(fk, fn, "rand" if dname == "randomized" else "plain", e / 255) >= THR).mean())
    res[dname] = r; print(dname, json.dumps(r, indent=1), flush=True)
json.dump(res, open(os.path.join(OUT, f"randomized_defense_{NAME}.json"), "w"), indent=2)
