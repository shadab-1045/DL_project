import os
import cv2
import numpy as np
from collections import defaultdict
from src.runtime.identity_pipeline import IdentityPipeline

def compute_metrics(genuine_scores, imposter_scores, threshold):
    # Genuine: score >= threshold -> True Accept, score < threshold -> False Reject
    ta = sum(1 for s in genuine_scores if s >= threshold)
    fr = sum(1 for s in genuine_scores if s < threshold)
    
    # Imposter: score >= threshold -> False Accept, score < threshold -> True Reject
    fa = sum(1 for s in imposter_scores if s >= threshold)
    tr = sum(1 for s in imposter_scores if s < threshold)
    
    tar = ta / len(genuine_scores) if genuine_scores else 0.0
    frr = fr / len(genuine_scores) if genuine_scores else 0.0
    far = fa / len(imposter_scores) if imposter_scores else 0.0
    trr = tr / len(imposter_scores) if imposter_scores else 0.0
    
    return tar, far, frr

def calibrate_and_evaluate():
    # Force threshold temporarily to allow identify() to return raw scores
    pipeline = IdentityPipeline()
    pipeline.set_threshold(0.0)
    
    base_dir = "sample_data/lfw"
    if not os.path.exists(base_dir):
        print("LFW dataset not found. Run download_lfw_subset.py first.")
        return

    identities = os.listdir(base_dir)
    print(f"Found identities: {identities}")

    # Data split: 0-4 for enrollment, 5-7 for validation, 8-9 for testing
    val_probes = []
    test_probes = []

    # 1. Enroll
    print("\n--- ENROLLMENT ---")
    for identity in identities:
        person_dir = os.path.join(base_dir, identity)
        images = sorted(os.listdir(person_dir))
        
        enroll_imgs = [os.path.join(person_dir, img) for img in images[:5]]
        pipeline.enroll(identity, enroll_imgs)
        
        for img in images[5:8]:
            val_probes.append((identity, os.path.join(person_dir, img)))
        for img in images[8:10]:
            test_probes.append((identity, os.path.join(person_dir, img)))

    print(f"\nEnrollment complete. Validation probes: {len(val_probes)}, Test probes: {len(test_probes)}")

    # Helper to get all similarities against gallery
    def get_all_similarities(img_path):
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        faces = pipeline.engine.extract_faces(img)
        if not faces:
            return None
        best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        live_embedding = best_face.normed_embedding

        scores = {}
        for enrolled_id, enrolled_emb in pipeline.gallery.identities.items():
            scores[enrolled_id] = pipeline.engine.compute_similarity(live_embedding, enrolled_emb)
        return scores

    # 2. Validation / Calibration
    print("\n--- THRESHOLD CALIBRATION (VALIDATION SET) ---")
    val_genuine_scores = []
    val_imposter_scores = []

    for true_id, img_path in val_probes:
        scores = get_all_similarities(img_path)
        if scores is None:
            continue
            
        for enrolled_id, sim in scores.items():
            if enrolled_id == true_id:
                val_genuine_scores.append(sim)
            else:
                val_imposter_scores.append(sim)

    print(f"Genuine pairs: {len(val_genuine_scores)}")
    print(f"Imposter pairs: {len(val_imposter_scores)}")
    
    if val_genuine_scores:
        print(f"Genuine similarity: Mean={np.mean(val_genuine_scores):.4f}, Min={np.min(val_genuine_scores):.4f}")
    if val_imposter_scores:
        print(f"Imposter similarity: Mean={np.mean(val_imposter_scores):.4f}, Max={np.max(val_imposter_scores):.4f}")

    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.45
    best_far_target = 0.01 # Aim for <= 1% FAR
    
    print("\nEvaluating Thresholds:")
    print("Thresh |   TAR   |   FAR   |   FRR   ")
    print("-------------------------------------")
    
    candidate_thresholds = []
    
    for t in thresholds:
        tar, far, frr = compute_metrics(val_genuine_scores, val_imposter_scores, t)
        print(f" {t:.2f}  | {tar:.4f} | {far:.4f} | {frr:.4f}")
        
        if far <= best_far_target:
            candidate_thresholds.append((t, tar, far, frr))

    if candidate_thresholds:
        # Pick the lowest threshold that satisfies the FAR condition to maximize TAR
        best_threshold = candidate_thresholds[0][0]
        print(f"\nSelected Candidate Threshold: {best_threshold:.2f} (FAR={candidate_thresholds[0][2]:.4f})")
    else:
        print(f"\nCould not strictly satisfy FAR <= {best_far_target}. Using provisional threshold 0.50")
        best_threshold = 0.50

    # 3. Final Evaluation on Test Set
    print("\n--- FINAL EVALUATION (HELD-OUT TEST SET) ---")
    pipeline.set_threshold(best_threshold)
    
    test_genuine_scores = []
    test_imposter_scores = []

    metrics = {"correct": 0, "total": 0, "false_accepts": 0, "false_rejects": 0}

    for true_id, img_path in test_probes:
        scores = get_all_similarities(img_path)
        if scores is None:
            continue
            
        for enrolled_id, sim in scores.items():
            if enrolled_id == true_id:
                test_genuine_scores.append(sim)
            else:
                test_imposter_scores.append(sim)
                
        # Simulate pipeline identification
        img = cv2.imread(img_path)
        res = pipeline.identify(img)
        match_id = res["identity"]
        
        metrics["total"] += 1
        if match_id == true_id:
            metrics["correct"] += 1
        else:
            if match_id is not None:
                metrics["false_accepts"] += 1
            else:
                metrics["false_rejects"] += 1

    tar, far, frr = compute_metrics(test_genuine_scores, test_imposter_scores, best_threshold)
    
    print(f"Test Genuine pairs: {len(test_genuine_scores)}")
    print(f"Test Imposter pairs: {len(test_imposter_scores)}")
    print(f"Threshold used: {best_threshold:.2f}")
    print(f"Test TAR: {tar:.4f}")
    print(f"Test FAR: {far:.4f}")
    print(f"Test FRR: {frr:.4f}")
    print(f"Pipeline Verification Accuracy: {metrics['correct']/metrics['total']:.4f}" if metrics['total'] else "Pipeline Verification Accuracy: N/A")

if __name__ == "__main__":
    calibrate_and_evaluate()
