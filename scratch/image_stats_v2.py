import os
import csv
import cv2
import numpy as np
from collections import defaultdict
from scipy.ndimage import center_of_mass

def get_stats():
    manifest_path = "data/manifests/train_pairs_v2.csv"
    with open(manifest_path, 'r') as f:
        pairs = list(csv.DictReader(f))
        
    stats_by_class = defaultdict(list)
    
    for pair in pairs:
        label = pair["label"]
        img = cv2.imread(pair["probe_path"])
        if img is None:
            continue
            
        # Per-channel mean/std
        b, g, r = cv2.split(img)
        ch_means = (b.mean(), g.mean(), r.mean())
        
        # Grayscale mean
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_mean = gray.mean()
        
        # Edge magnitude center of mass
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        com_y, com_x = center_of_mass(magnitude)
        
        stats_by_class[label].append({
            "b_mean": ch_means[0],
            "g_mean": ch_means[1],
            "r_mean": ch_means[2],
            "gray_mean": gray_mean,
            "edge_com_y": com_y,
            "edge_com_x": com_x,
            "size": os.path.getsize(pair["probe_path"])
        })
        
    for label, metrics_list in stats_by_class.items():
        print(f"\n--- {label.upper()} ---")
        n = len(metrics_list)
        b = np.nanmean([m["b_mean"] for m in metrics_list])
        g = np.nanmean([m["g_mean"] for m in metrics_list])
        r = np.nanmean([m["r_mean"] for m in metrics_list])
        gray = np.nanmean([m["gray_mean"] for m in metrics_list])
        com_y = np.nanmean([m["edge_com_y"] for m in metrics_list])
        com_x = np.nanmean([m["edge_com_x"] for m in metrics_list])
        size = np.nanmean([m["size"] for m in metrics_list])
        print(f"N = {n}")
        print(f"Mean BGR: ({b:.2f}, {g:.2f}, {r:.2f})")
        print(f"Mean Grayscale: {gray:.2f}")
        print(f"Mean Edge Center-of-Mass (Y, X): ({com_y:.2f}, {com_x:.2f})")
        print(f"Mean File Size (bytes): {size:.2f}")

if __name__ == "__main__":
    get_stats()
