import cv2
import numpy as np

def get_context_crop(image_bgr, bbox, context_margin=0.5):
    """
    Extracts a face crop with additional context around the bounding box.
    bbox is [x1, y1, x2, y2].
    context_margin is the ratio of width/height to add as padding.
    """
    h_img, w_img = image_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    
    w_box = x2 - x1
    h_box = y2 - y1
    
    pad_w = int(w_box * context_margin)
    pad_h = int(h_box * context_margin)
    
    new_x1 = max(0, x1 - pad_w)
    new_y1 = max(0, y1 - pad_h)
    new_x2 = min(w_img, x2 + pad_w)
    new_y2 = min(h_img, y2 + pad_h)
    
    crop = image_bgr[new_y1:new_y2, new_x1:new_x2]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
