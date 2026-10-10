"""Adversarial evaluation of FaceGuard.
Exp A: InSwapper face-swap attack -> does FaceGuard flag the swapped faces?
Exp B: FGSM / PGD (L-inf) evasion attack on known-fake frames -> how fast does detection collapse?
Usage: python eval_adversarial.py [N]"""
import os, sys, json, glob, random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np, cv2, tensorflow as tf

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "eval_data", "ffpp_c32", "Frames(cropped+aligned)")
MODEL = os.path.join(ROOT, "FaceGuard-Digital-Forensic-System", "models", "faceguard_phase2_finetuned.h5")
SWAPPER = os.path.join(os.path.dirname(ROOT), "models", "inswapper_128.onnx")
OUT = os.path.join(ROOT, "eval_results"); os.makedirs(OUT, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
T = 0.40
rng = random.Random(7)
model = tf.keras.models.load_model(MODEL)
res = {"n": N, "threshold": T}

def load_rgb(p): return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
def prep(rgb): return cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
def score(batch): return model.predict(np.stack(batch), verbose=0).ravel()

# ---------------- Exp A: InSwapper attack ----------------
from insightface.app import FaceAnalysis
import insightface
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"]); fa.prepare(ctx_id=-1, det_size=(320, 320))
swapper = insightface.model_zoo.get_model(SWAPPER, download=False, download_zip=False)
reals = sorted(glob.glob(os.path.join(DATA, "Original", "*.jpg"))); rng.shuffle(reals)
orig, swapped = [], []
for i in range(0, len(reals) - 1, 2):
    if len(orig) >= N: break
    # upscale small crops so the detector finds the face
    tgt = cv2.resize(cv2.imread(reals[i]), (512, 512)); src = cv2.resize(cv2.imread(reals[i + 1]), (512, 512))
    ft, fs = fa.get(tgt), fa.get(src)
    if not ft or not fs: continue
    out = swapper.get(tgt.copy(), ft[0], fs[0], paste_back=True)
    orig.append(prep(cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB))); swapped.append(prep(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)))
    if len(orig) == 1: cv2.imwrite(os.path.join(OUT, "swap_example.jpg"), np.hstack([tgt, src, out]))
so, ss = score(orig), score(swapped)
res["swap_attack"] = {"n_pairs": len(orig), "real_flagged_fake_rate(FPR)": float((so > T).mean()),
                      "swapped_flagged_fake_rate(detection)": float((ss > T).mean()),
                      "mean_score_real": float(so.mean()), "mean_score_swapped": float(ss.mean()),
                      "swap_evades_detection_rate": float((ss <= T).mean())}
print("swap:", res["swap_attack"])

# ---------------- Exp B: FGSM / PGD evasion ----------------
fakes = []
for m in ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]:
    fakes += rng.sample(sorted(glob.glob(os.path.join(DATA, m, "*.jpg"))), N // 5)
X = np.stack([prep(load_rgb(p)) for p in fakes])
clean = score(list(X)); keep = clean > T            # attack only fakes FaceGuard catches when clean
X = X[keep]; res["adv_start_n_caught_fakes"] = int(keep.sum())

def attack(x, eps, steps):
    x = tf.constant(x); adv = tf.identity(x); a = eps if steps == 1 else 2.5 * eps / steps
    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(adv); p = model(adv, training=False)
            loss = tf.reduce_mean(tf.math.log(p + 1e-7))   # minimise P(fake)
        g = tape.gradient(loss, adv)
        adv = adv - a * tf.sign(g)
        adv = tf.clip_by_value(tf.clip_by_value(adv, x - eps, x + eps), 0, 1)
    return adv.numpy()

res["evasion"] = {}
for name, steps in [("FGSM", 1), ("PGD-10", 10)]:
    for e255 in [0, 1, 2, 4, 8]:
        eps = e255 / 255
        s = np.concatenate([model.predict(attack(X[i:i + 32], eps, steps) if e255 else X[i:i + 32], verbose=0).ravel()
                            for i in range(0, len(X), 32)])
        res["evasion"][f"{name} eps={e255}/255"] = {"detection_rate": float((s > T).mean()), "attack_success": float((s <= T).mean())}
        print(name, e255, res["evasion"][f"{name} eps={e255}/255"])
json.dump(res, open(os.path.join(OUT, "faceguard_adversarial.json"), "w"), indent=2)
