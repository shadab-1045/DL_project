import os
import cv2
import json
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

def get_face(app, img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read {img_path}")
    faces = app.get(img)
    if not faces:
        raise ValueError(f"No faces found in {img_path}")
    return img, faces[0]

def main():
    print("Loading buffalo_l...")
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    print("Loading inswapper_128...")
    swapper = get_model('models/inswapper_128.onnx', download=False)
    
    # We use sample images from vgg_face2-master if they exist
    src_path = r"..\vgg_face2-master\samples (test set)\loose_crop (release version)\n000106\0003_01.jpg"
    dst_path = r"..\vgg_face2-master\samples (test set)\loose_crop (release version)\n007241\0005_01.jpg"
    
    if not os.path.exists(src_path) or not os.path.exists(dst_path):
        print("Sample images not found. Smoke test requires sample images.")
        return
        
    print("Detecting faces...")
    src_img, src_face = get_face(app, src_path)
    dst_img, dst_face = get_face(app, dst_path)
    
    print("Performing face swap...")
    # swapper.get takes (img, target_face, source_face, paste_back=True)
    res_img = swapper.get(dst_img, dst_face, src_face, paste_back=True)
    
    out_dir = "data/generated/face_swaps/smoke_test"
    os.makedirs(out_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(out_dir, "source.jpg"), src_img)
    cv2.imwrite(os.path.join(out_dir, "target.jpg"), dst_img)
    cv2.imwrite(os.path.join(out_dir, "swap.jpg"), res_img)
    
    metadata = {
        "reference_identity": "n000106",
        "visible_identity": "n000106",
        "physical_identity": "n007241",
        "label": "impersonation",
        "attack_type": "face_swap",
        "generator": "inswapper"
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    # Create contact sheet
    import numpy as np
    # Resize them to same height
    h = 256
    def resize_img(img):
        ratio = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * ratio), h))
        
    s = resize_img(src_img)
    t = resize_img(dst_img)
    r = resize_img(res_img)
    contact = np.hstack([s, t, r])
    cv2.imwrite(os.path.join(out_dir, "contact_sheet.jpg"), contact)
    
    print("Smoke test completed successfully. Outputs in", out_dir)

if __name__ == '__main__':
    main()
