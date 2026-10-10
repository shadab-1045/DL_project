"""Build the swap-detector dataset with the project's real attacker (InSwapper).
For each target face: real crop (untouched) and fake crop (another identity swapped in). Both use the SAME bbox
from the same detector, so crop geometry can't leak the label. Split is by identity (LFW person / FF++ video id).
Usage: python gen_swap_data.py [N_LFW] [N_FFPP]   -> data/swapdet/{train,test}/{real,fake}/*.jpg + meta.csv"""
import os, sys, glob, random, csv
import numpy as np, cv2
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data", "swapdet")
SWAPPER = os.path.join(os.path.dirname(ROOT), "models", "inswapper_128.onnx")
FFPP = os.path.join(ROOT, "eval_data", "ffpp_c32", "Frames(cropped+aligned)", "Original")
N_LFW = int(sys.argv[1]) if len(sys.argv) > 1 else 2400
N_FF = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
UP = 384

def crop(img, bbox, m=0.35, out=224):
    x1, y1, x2, y2 = bbox; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; s = max(x2 - x1, y2 - y1) * (1 + m)
    x1, y1 = int(cx - s / 2), int(cy - s / 2); s = int(s)
    pad = s; big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return cv2.resize(big[y1 + pad:y1 + pad + s, x1 + pad:x1 + pad + s], (out, out), interpolation=cv2.INTER_AREA)

_fa = _sw = None
def init():
    global _fa, _sw
    import onnxruntime as ort, insightface
    from insightface.app import FaceAnalysis
    _orig = ort.InferenceSession.__init__   # cap threads per session: N workers x all-cores thrashes badly
    def _capped(self, path, sess_options=None, *a, **kw):
        so = sess_options or ort.SessionOptions(); so.intra_op_num_threads = 2; so.inter_op_num_threads = 1
        _orig(self, path, sess_options=so, *a, **kw)
    ort.InferenceSession.__init__ = _capped
    _fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "recognition"])
    _fa.prepare(ctx_id=-1, det_size=(320, 320))
    _sw = insightface.model_zoo.get_model(SWAPPER, download=False, download_zip=False)

def work(job):
    split, tag, tgt, src = job
    t = cv2.resize(tgt, (UP, UP)); s = cv2.resize(src, (UP, UP))
    ft, fs = _fa.get(t), _fa.get(s)
    if not ft or not fs: return None
    ft = max(ft, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])); fs = max(fs, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    fake = _sw.get(t.copy(), ft, fs, paste_back=True)
    # also record how well the swap fools ArcFace: sim(fake, source identity) vs sim(fake, target identity)
    ff = _fa.get(fake); sim_src = sim_tgt = -1.0
    if ff:
        e = max(ff, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])).normed_embedding
        sim_src, sim_tgt = float(e @ fs.normed_embedding), float(e @ ft.normed_embedding)
    for cls, im in (("real", t), ("fake", fake)):
        d = os.path.join(OUT, split, cls); cv2.imwrite(os.path.join(d, f"{tag}.jpg"), crop(im, ft.bbox), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return (split, tag, sim_src, sim_tgt)

def main():
    rng = random.Random(0)
    for sp in ("train", "test"):
        for c in ("real", "fake"): os.makedirs(os.path.join(OUT, sp, c), exist_ok=True)
    from sklearn.datasets import fetch_lfw_people
    d = fetch_lfw_people(color=True, slice_=None, resize=1.0, min_faces_per_person=2)
    imgs = [cv2.cvtColor((i if i.max() > 1.5 else i * 255).astype(np.uint8), cv2.COLOR_RGB2BGR) for i in d.images]
    ids = d.target; people = sorted(set(ids)); rng.shuffle(people); test_people = set(people[: len(people) // 6])
    pool = {"train": [], "test": []}
    for i, (im, p) in enumerate(zip(imgs, ids)): pool["test" if p in test_people else "train"].append((f"lfw{i}", im))
    ff = sorted(glob.glob(os.path.join(FFPP, "*.jpg"))); vids = sorted({int(os.path.basename(f).split("_")[0]) for f in ff})
    test_v = set(rng.sample(vids, len(vids) // 6)); ffp = {"train": [], "test": []}
    for f in rng.sample(ff, min(len(ff), 6000)):
        ffp["test" if int(os.path.basename(f).split("_")[0]) in test_v else "train"].append((f"ff{os.path.basename(f)[:-4]}", cv2.imread(f)))
    jobs = []
    for sp in ("train", "test"):
        for src_pool, n in ((pool[sp], int(N_LFW * (0.85 if sp == "train" else 0.15))), (ffp[sp], int(N_FF * (0.85 if sp == "train" else 0.15)))):
            tg = rng.sample(src_pool, min(n, len(src_pool)))
            for tag, im in tg:
                stag, sim = rng.choice(src_pool)
                if stag != tag: jobs.append((sp, tag, im, sim))
    done = lambda sp, tag: all(os.path.exists(os.path.join(OUT, sp, c, f"{tag}.jpg")) for c in ("real", "fake"))
    jobs = [j for j in jobs if not done(j[0], j[1])]; random.Random(1).shuffle(jobs)   # resumable; any prefix has both splits
    print("jobs", len(jobs), flush=True)
    mp = os.path.join(OUT, "meta.csv"); new_file = not os.path.exists(mp); n = 0
    with open(mp, "a", newline="") as f, Pool(10, initializer=init) as p:
        w = csv.writer(f)
        if new_file: w.writerow(["split", "tag", "sim_to_source", "sim_to_target"])
        for k, r in enumerate(p.imap_unordered(work, jobs, chunksize=2)):
            if r: w.writerow(r); f.flush(); n += 1
            if k % 100 == 0: print(k, n, flush=True)
    print("done", n)

if __name__ == "__main__":
    main()
