import os
import cv2
import numpy as np

# Use sklearn to fetch LFW dataset to avoid manual downloading of large tarballs
# It requires scikit-learn to be installed. We'll install it dynamically if needed.
try:
    from sklearn.datasets import fetch_lfw_people
except ImportError:
    import subprocess
    import sys
    print("Installing scikit-learn for LFW dataset...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.datasets import fetch_lfw_people

def create_lfw_subset():
    print("Fetching LFW subset (min_faces_per_person=20)...")
    # Using min_faces_per_person=20 ensures we get people with enough images for enrollment + probes
    lfw = fetch_lfw_people(min_faces_per_person=20, color=True, slice_=None)
    
    base_dir = "sample_data/lfw"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    target_names = lfw.target_names
    images = lfw.images
    targets = lfw.target

    # Select 10 identities for our test
    selected_identities = target_names[:10]
    print(f"Selected identities: {selected_identities}")

    identity_counts = {name: 0 for name in selected_identities}

    for img, target in zip(images, targets):
        name = target_names[target]
        if name in selected_identities:
            count = identity_counts[name]
            # Save up to 10 images per person
            if count < 10:
                person_dir = os.path.join(base_dir, name.replace(" ", "_"))
                os.makedirs(person_dir, exist_ok=True)
                
                # Convert from RGB (sklearn format) [0..1] floats or [0..255] ints to BGR uint8
                if img.dtype != np.uint8:
                    img_uint8 = (img * 255).astype(np.uint8)
                else:
                    img_uint8 = img
                
                img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
                
                # Resize slightly because LFW is small (250x250 default, or sliced)
                # fetch_lfw_people returns original size if slice_=None
                
                img_path = os.path.join(person_dir, f"{count:02d}.jpg")
                cv2.imwrite(img_path, img_bgr)
                
                identity_counts[name] += 1

    print("LFW subset created successfully.")

if __name__ == "__main__":
    create_lfw_subset()
