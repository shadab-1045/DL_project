"""Evaluate FaceGuard on FaceForensics++ frames. Usage: python eval_faceguard.py [N_REAL] [MODEL.h5]"""
import os, sys, json, random, glob
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_fscore_support

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "eval_data", "ffpp_c32", "Frames(cropped+aligned)")
N_REAL = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
MODEL = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "FaceGuard-Digital-Forensic-System", "models", "faceguard_phase2_finetuned.h5")
OUT = os.path.join(ROOT, "eval_results"); os.makedirs(OUT, exist_ok=True)
THRESH = 0.40  # same threshold the app uses (score > 0.40 => FAKE)
METHODS = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]

rng = random.Random(42)
def sample(m, n):
    fs = sorted(glob.glob(os.path.join(DATA, m, "*.jpg")))
    return rng.sample(fs, min(n, len(fs)))

items = [(f, 0, "Original") for f in sample("Original", N_REAL)]
for m in METHODS:  # fakes balanced across methods so total fake == total real
    items += [(f, 1, m) for f in sample(m, N_REAL // len(METHODS))]

from tensorflow.keras.models import load_model
model = load_model(MODEL)
print("model:", MODEL, "input:", model.input_shape, "n:", len(items))

def prep(path):  # identical to FaceGuard backend/model.py preprocess_frame
    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0

scores = []
for i in range(0, len(items), 64):
    batch = np.stack([prep(p) for p, _, _ in items[i:i + 64]])
    scores += model.predict(batch, verbose=0).ravel().tolist()
    print(f"\r{len(scores)}/{len(items)}", end="")
print()

y = np.array([l for _, l, _ in items]); s = np.array(scores); meth = np.array([m for _, _, m in items])
pred = (s > THRESH).astype(int)
p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
cm = confusion_matrix(y, pred)
res = {"model": os.path.basename(MODEL), "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()),
       "threshold": THRESH, "accuracy": float((pred == y).mean()), "precision": float(p), "recall": float(r),
       "f1": float(f1), "auc": float(roc_auc_score(y, s)), "confusion_matrix[[TN,FP],[FN,TP]]": cm.tolist(),
       "per_method_detection_rate": {m: float((pred[meth == m] == 1).mean()) for m in METHODS},
       "real_correctly_accepted": float((pred[meth == "Original"] == 0).mean())}
json.dump(res, open(os.path.join(OUT, f"faceguard_ffpp_metrics_{os.path.basename(MODEL)[:-3]}.json"), "w"), indent=2)
print(json.dumps(res, indent=2))

fpr, tpr, _ = roc_curve(y, s)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].plot(fpr, tpr, label=f"AUC={res['auc']:.3f}"); ax[0].plot([0, 1], [0, 1], "k--")
ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="FaceGuard ROC (FF++ C32)"); ax[0].legend()
ax[1].imshow(cm, cmap="Blues"); ax[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"],
                                           xlabel="Predicted", ylabel="Actual", title=f"Confusion @ {THRESH}")
for (a, b), v in np.ndenumerate(cm): ax[1].text(b, a, v, ha="center", va="center", color="red", fontsize=14)
plt.tight_layout(); plt.savefig(os.path.join(OUT, f"faceguard_ffpp_roc_confusion_{os.path.basename(MODEL)[:-3]}.png"), dpi=130)
