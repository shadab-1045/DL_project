import os
import cv2
from src.runtime.identity_pipeline import IdentityPipeline

def evaluate():
    pipeline = IdentityPipeline()

    base_dir = "sample_data"
    
    # 1. Enroll
    print("\n--- ENROLLMENT ---")
    pipeline.enroll("P001", [f"{base_dir}/P001_ref1.jpg", f"{base_dir}/P001_ref2.jpg"])
    pipeline.enroll("P002", [f"{base_dir}/P002_ref1.jpg"])

    # 2. Evaluation
    print("\n--- OFFLINE EVALUATION ---")
    
    test_cases = [
        {"name": "P001_probe (Genuine)", "path": f"{base_dir}/P001_probe.jpg", "expected": "P001"},
        {"name": "P002_probe (Genuine)", "path": f"{base_dir}/P002_probe.jpg", "expected": "P002"},
        {"name": "P001_ref1 vs Gallery (Same)", "path": f"{base_dir}/P001_ref1.jpg", "expected": "P001"},
    ]

    metrics = {
        "total": len(test_cases),
        "correct": 0,
        "false_accepts": 0,
        "false_rejects": 0
    }

    for case in test_cases:
        img = cv2.imread(case["path"])
        if img is None:
            print(f"Skipping {case['name']} - image not found")
            continue
            
        res = pipeline.identify(img)
        match_id = res["identity"]
        
        print(f"Test {case['name']}:")
        print(f"  Expected: {case['expected']} | Got: {match_id}")
        print(f"  Similarity: {res['similarity']:.4f} | Status: {res['status']}")
        
        if case["expected"] == match_id:
            metrics["correct"] += 1
            print("  -> PASS")
        else:
            if match_id is not None and match_id != case["expected"]:
                metrics["false_accepts"] += 1
            if match_id is None and case["expected"] is not None:
                metrics["false_rejects"] += 1
            print("  -> FAIL")

    print("\n--- METRICS ---")
    print(f"Accuracy: {metrics['correct']/metrics['total']:.2f}")
    print(f"False Accepts: {metrics['false_accepts']}")
    print(f"False Rejects: {metrics['false_rejects']}")

if __name__ == "__main__":
    evaluate()
