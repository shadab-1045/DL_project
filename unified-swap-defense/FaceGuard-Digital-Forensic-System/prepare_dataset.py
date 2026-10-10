import os
import random
import shutil
import uuid
from glob import glob

# ================= CONFIGURATION =================
DATASET_ROOT = "FaceForensics++"   # Your main folder
OUTPUT_ROOT = "data"               # Where the new dataset goes

TOTAL_SAMPLES = 25000              # Total images (Real + Fake)
TRAIN_RATIO = 0.8                  # 80% Train, 20% Val
RANDOM_SEED = 42

# Look for these file types (Case insensitive in this script)
VALID_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg")
# =================================================


def make_dirs():
    """Create the train/val and real/fake directory structure."""
    if os.path.exists(OUTPUT_ROOT):
        print(f"⚠️  Warning: '{OUTPUT_ROOT}' folder already exists. Merging files...")
    
    for split in ["train", "val"]:
        for cls in ["real", "fake"]:
            os.makedirs(os.path.join(OUTPUT_ROOT, split, cls), exist_ok=True)


def get_files_recursive(folder_path):
    """
    Recursively find all images in a folder, matching VALID_EXTENSIONS.
    """
    found_files = []
    # Loop through extensions (.png, .jpg, etc)
    for ext in VALID_EXTENSIONS:
        # ** means 'look in all subfolders', recursive=True turns that feature on
        pattern = os.path.join(folder_path, "**", ext)
        found_files.extend(glob(pattern, recursive=True))
    return found_files


def collect_fake_images():
    """
    Collect fake images and keep track of which method (Deepfakes, etc.) they came from.
    """
    fake_images = []
    fake_root = os.path.join(DATASET_ROOT, "fake")

    if not os.path.exists(fake_root):
        print(f"❌ Error: Could not find folder: {os.path.abspath(fake_root)}")
        return []

    # Iterate over method folders (Deepfakes, FaceSwap, etc.)
    for method in os.listdir(fake_root):
        method_path = os.path.join(fake_root, method)
        
        if os.path.isdir(method_path):
            images = get_files_recursive(method_path)
            # Store as tuple: (full_path, method_name)
            for img in images:
                fake_images.append((img, method))
                
    return fake_images


def safe_copy(src, dst):
    """
    Try to create a link (saves space). Fall back to copy if linking fails.
    """
    try:
        if os.path.exists(dst):
            os.remove(dst) # Remove existing file to avoid error
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def process_batch(file_list, split_type, class_name, is_fake=False):
    """
    Process a list of files: copy them to the destination with a unique name.
    """
    for item in file_list:
        if is_fake:
            img_path, method = item
            # e.g., "Deepfakes_1a2b3c_image01.png"
            unique_name = f"{method}_{uuid.uuid4().hex[:8]}_{os.path.basename(img_path)}"
        else:
            img_path = item
            # e.g., "real_1a2b3c_image01.png"
            unique_name = f"real_{uuid.uuid4().hex[:8]}_{os.path.basename(img_path)}"

        dst = os.path.join(OUTPUT_ROOT, split_type, class_name, unique_name)
        safe_copy(img_path, dst)


def main():
    print(f"🚀 Starting dataset preparation...")
    print(f"📂 Looking for data in: {os.path.abspath(DATASET_ROOT)}")
    
    random.seed(RANDOM_SEED)
    make_dirs()

    # 1. Collect Images
    real_path_root = os.path.join(DATASET_ROOT, "real")
    if not os.path.exists(real_path_root):
        print(f"❌ Error: Could not find folder: {os.path.abspath(real_path_root)}")
        return

    real_imgs = get_files_recursive(real_path_root)
    fake_imgs = collect_fake_images()

    # 2. Debug Prints (The "Sanity Check")
    print(f"📊 Found Real Images: {len(real_imgs)}")
    print(f"📊 Found Fake Images: {len(fake_imgs)}")

    if len(real_imgs) == 0 or len(fake_imgs) == 0:
        print("\n❌ CRITICAL ERROR: No images found!")
        print("   - Check if your folder names match 'real' and 'fake' exactly.")
        print("   - Check if the images are actually inside subfolders.")
        return

    # 3. Validation
    half = TOTAL_SAMPLES // 2
    if len(real_imgs) < half:
        print(f"⚠️  Not enough REAL images. Needed {half}, found {len(real_imgs)}. Using all available.")
        half = min(len(real_imgs), len(fake_imgs)) # Adjust to smallest class
    
    # 4. Shuffle & Slice
    random.shuffle(real_imgs)
    random.shuffle(fake_imgs)

    real_imgs = real_imgs[:half]
    fake_imgs = fake_imgs[:half]

    print(f"✂️  Sampling {len(real_imgs)} Real and {len(fake_imgs)} Fake images...")

    # 5. Split Train/Val
    split_r = int(len(real_imgs) * TRAIN_RATIO)
    split_f = int(len(fake_imgs) * TRAIN_RATIO)

    # Process Real
    process_batch(real_imgs[:split_r], "train", "real", is_fake=False)
    process_batch(real_imgs[split_r:], "val", "real", is_fake=False)

    # Process Fake
    process_batch(fake_imgs[:split_f], "train", "fake", is_fake=True)
    process_batch(fake_imgs[split_f:], "val", "fake", is_fake=True)

    print("\n✅ DONE! Dataset is ready.")
    print(f"   Train: {os.path.join(OUTPUT_ROOT, 'train')}")
    print(f"   Val:   {os.path.join(OUTPUT_ROOT, 'val')}")


if __name__ == "__main__":
    main()