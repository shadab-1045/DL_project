"""Train the InSwapper-swap detector (MobileNetV2, GPU) twice: clean control vs PGD adversarial training.
Evaluate both under white-box FGSM/PGD. Usage: python train_swapdet.py [EPOCHS]"""
import os, sys, glob, json, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # 8 GB GPU shared with the desktop
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from swapdet import Net
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "swapdet"); OUT = os.path.join(ROOT, "eval_results"); MODELS = os.path.join(ROOT, "models"); os.makedirs(MODELS, exist_ok=True)
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
EPS255 = float(sys.argv[2]) if len(sys.argv) > 2 else None   # `python train_swapdet.py 10 1` -> only the adversarial run at eps=1/255
dev = "cuda"; TRAIN_EPS = (EPS255 or 4) / 255; TRAIN_STEPS = 5
ADV_W = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5   # weight of the adversarial loss term (rest is clean loss)
torch.manual_seed(0); np.random.seed(0); random.seed(0)

def augment(im):  # webcam-like degradations so the detector isn't keyed on pristine-pixel artifacts
    if random.random() < .5: im = im[:, ::-1]
    if random.random() < .5:
        s = random.choice([96, 128, 160]); im = cv2.resize(cv2.resize(im, (s, s), interpolation=cv2.INTER_AREA), (224, 224))
    if random.random() < .3: im = cv2.GaussianBlur(im, (0, 0), random.uniform(.4, 1.5))
    if random.random() < .6:
        im = cv2.imdecode(cv2.imencode(".jpg", np.ascontiguousarray(im), [cv2.IMWRITE_JPEG_QUALITY, random.randint(35, 95)])[1], 1)
    im = np.clip(im.astype(np.float32) * random.uniform(.75, 1.25) + random.uniform(-20, 20), 0, 255)
    if random.random() < .3: im = np.clip(im + np.random.randn(*im.shape) * random.uniform(2, 8), 0, 255)
    return im.astype(np.uint8)

class DS(Dataset):
    def __init__(s, split, aug): s.aug = aug; s.items = [(f, c) for c, k in enumerate(("real", "fake")) for f in sorted(glob.glob(os.path.join(DATA, split, k, "*.jpg")))]
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        f, c = s.items[i]; im = cv2.imread(f)
        if s.aug: im = augment(im)
        return torch.from_numpy(cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255, torch.tensor(float(c))

def pgd(model, x, y, eps, steps, rand=True):
    modes = [m.training for m in model.modules()]; model.eval(); a = 2.5 * eps / steps if steps > 1 else eps
    adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1) if rand else x.clone()
    for _ in range(steps):
        adv.requires_grad_(True); loss = F.binary_cross_entropy_with_logits(model(adv), y)
        g, = torch.autograd.grad(loss, adv); adv = (adv.detach() + a * g.sign()); adv = torch.min(torch.max(adv, x - eps), x + eps).clamp(0, 1)
    for m, t in zip(model.modules(), modes): m.training = t   # restore each module's own mode (keeps frozen BN frozen)
    return adv.detach()

def margin_pgd(model, x, eps, steps=50, restarts=2):
    """Strong untargeted attack on swapped faces: PGD on the raw logit (no sigmoid saturation => no gradient masking), random restarts,
    worst case per sample. Authoritative robustness metric; BCE-PGD overstates robustness (see eval_results/margin_attack_results.json)."""
    best = torch.full((len(x),), 1e9, device=x.device); best_adv = x.clone(); a = 2.5 * eps / steps
    for _ in range(restarts):
        adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
        for _ in range(steps):
            adv.requires_grad_(True); g, = torch.autograd.grad(model(adv).sum(), adv)
            adv = torch.min(torch.max(adv.detach() - a * g.sign(), x - eps), x + eps).clamp(0, 1)
        with torch.no_grad(): lg = model(adv)
        m = lg < best; best[m] = lg[m]; best_adv[m] = adv[m]
    return best_adv

def freeze_bn(model):  # attacks are crafted in eval mode; keep BN stats fixed so train/attack/eval all see the same function
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d): m.eval()

def train(adversarial, init=None):
    """init=None: full training from ImageNet weights (BN in train mode). init=ckpt: fine-tune that checkpoint with BN statistics
    frozen (so the attacked function == the trained function), lower LR. adversarial=True adds a PGD loss with eps ramped 0->TRAIN_EPS."""
    model = Net(pretrained=init is None).to(dev); lr = 3e-4 if init is None else 1e-4
    if init: model.load_state_dict(torch.load(init, map_location=dev, weights_only=True))
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-4)
    dl = DataLoader(DS("train", True), batch_size=32, shuffle=True, num_workers=4, drop_last=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=EPOCHS * len(dl)); step = 0; ramp = 3 * len(dl)
    for ep in range(EPOCHS):
        model.train()
        if init: freeze_bn(model)
        tot = 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev); loss = F.binary_cross_entropy_with_logits(model(x), y)
            if adversarial:
                eps = TRAIN_EPS * min(1.0, (step + 1) / ramp)
                loss = (1 - ADV_W) * loss + ADV_W * F.binary_cross_entropy_with_logits(model(pgd(model, x, y, eps, TRAIN_STEPS)), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item(); step += 1
        print(f"  [{'adv' if adversarial else 'clean'}{'-ft' if init else ''}] epoch {ep + 1}/{EPOCHS} loss {tot / len(dl):.4f}", flush=True)
    return model.eval()

def evaluate(model):
    dl = DataLoader(DS("test", False), batch_size=32, num_workers=2); xs, ys = [], []
    for x, y in dl: xs.append(x); ys.append(y)
    X, Y = torch.cat(xs), torch.cat(ys); ynp = Y.numpy()
    def scores(att):
        out = []
        for i in range(0, len(X), 32):
            x, y = X[i:i + 32].to(dev), Y[i:i + 32].to(dev); x = att(x, y) if att else x
            with torch.no_grad(): out.append(torch.sigmoid(model(x)).cpu())
        return torch.cat(out).numpy()
    s = scores(None); r = {"n_test": len(Y), "clean_acc": float(((s > .5) == ynp).mean()), "auc": float(roc_auc_score(ynp, s)),
                           "swap_detection_TPR": float((s[ynp == 1] > .5).mean()), "real_false_alarm_FPR": float((s[ynp == 0] > .5).mean())}
    for name, steps in (("FGSM", 1), ("PGD10", 10)):
        for e in (1, 2, 4, 8):
            sa = scores(lambda x, y: pgd(model, x, y, e / 255, steps)); r[f"{name}_eps{e}_robust_acc"] = float(((sa > .5) == ynp).mean())
            r[f"{name}_eps{e}_swap_detection"] = float((sa[ynp == 1] > .5).mean())
    sa = scores(lambda x, y: pgd(model, x, y, 4 / 255, 50)); r["PGD50_eps4_robust_acc"] = float(((sa > .5) == ynp).mean())
    return r

if __name__ == "__main__":
    base = os.path.join(MODELS, "swapdet_clean.pt"); rp = os.path.join(OUT, "swapdet_results.json")
    if EPS255:  # single adversarial run at a chosen training eps, merged into the results file
        res = json.load(open(rp)); name = f"adversarial_eps{EPS255:g}" + (f"_w{ADV_W:g}" if ADV_W != 0.5 else "")
        m = train(True, init=base); torch.save(m.state_dict(), os.path.join(MODELS, f"swapdet_{name}.pt")); res[name] = evaluate(m)
        res[name]["train_eps"] = f"{EPS255:g}/255"; res[name]["adv_loss_weight"] = ADV_W; json.dump(res, open(rp, "w"), indent=2); print(name, json.dumps(res[name], indent=1), flush=True); sys.exit()
    base = os.path.join(MODELS, "swapdet_clean.pt")
    res = {"epochs": EPOCHS, "protocol": "clean = full training (BN train mode); control & adversarial = EPOCHS of fine-tuning FROM clean with frozen BN, same LR/schedule; "
                                         "adversarial adds PGD-5 loss, eps=4/255 ramped over 3 epochs, 50% clean + 50% adversarial"}
    if not os.path.exists(base): torch.save(train(False).state_dict(), base)
    m = Net().to(dev).eval(); m.load_state_dict(torch.load(base, map_location=dev, weights_only=True)); res["clean"] = evaluate(m); print("clean", json.dumps(res["clean"]), flush=True)
    for name, adv in (("clean_finetune_control", False), ("adversarial", True)):
        m = train(adv, init=base); torch.save(m.state_dict(), os.path.join(MODELS, f"swapdet_{name}.pt")); res[name] = evaluate(m); print(name, json.dumps(res[name], indent=1), flush=True)
        json.dump(res, open(os.path.join(OUT, "swapdet_results.json"), "w"), indent=2)
    eps = [0, 1, 2, 4, 8]; plt.figure(figsize=(6.5, 4))
    for n in ("clean", "clean_finetune_control", "adversarial"): plt.plot(eps, [res[n]["clean_acc"]] + [res[n][f"PGD10_eps{e}_robust_acc"] for e in eps[1:]], "o-", label=n)
    plt.xlabel("PGD-10 epsilon (x/255)"); plt.ylabel("accuracy"); plt.title("Swap detector under white-box attack"); plt.ylim(0, 1); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "swapdet_robustness.png"), dpi=130)
