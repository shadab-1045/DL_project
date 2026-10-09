import cv2
import numpy as np
from insightface.utils import face_align

def align_face(image_bgr, kps, image_size=112):
    """
    Aligns and crops the face using 5 facial landmarks (kps).
    Returns the aligned RGB face crop of size (image_size, image_size).
    """
    # norm_crop expects BGR image and kps (5x2 array)
    aligned_face_bgr = face_align.norm_crop(image_bgr, landmark=kps, image_size=image_size)
    aligned_face_rgb = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB)
    return aligned_face_rgb
