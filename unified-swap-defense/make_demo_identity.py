"""Export a few LFW photos of one person to <repo root>/sample_data/lfw/<Name>/ so the server can enroll them as "Alice",
the identity the simulated face-swap attack impersonates. Usage: python make_demo_identity.py [Name]  (default Angelina_Jolie)"""
import os, sys, cv2, numpy as np
from sklearn.datasets import fetch_lfw_people

name = sys.argv[1] if len(sys.argv) > 1 else "Angelina_Jolie"
d = fetch_lfw_people(color=True, slice_=None, resize=1.0, min_faces_per_person=5)
names = [n.replace(" ", "_") for n in d.target_names]
if name not in names: sys.exit(f"{name} not in LFW (min 5 photos). Examples: {names[:8]}")
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data", "lfw", name); os.makedirs(out, exist_ok=True)
idx = np.where(d.target == names.index(name))[0][:5]
for k, i in enumerate(idx):
    im = d.images[i]; im = (im if im.max() > 1.5 else im * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out, f"{k}.jpg"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
print(f"wrote {len(idx)} images to {out}")
