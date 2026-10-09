import cv2
import io
from PIL import Image
from torchvision import transforms

class V4LiveParityAdapter:
    """
    Adapter to strictly reproduce the V4 dataset preprocessing path (Phase 7F.20).
    The V4 dataset (V2 manifest) generated aligned faces by saving them to disk as JPEGs
    using cv2.imwrite, and loading them with PIL.Image.open. This caused the model to
    learn JPEG compression artifacts as a feature of Genuine inputs.
    
    This adapter applies an identical in-memory encode/decode cycle to replicate 
    the exact JPEG compression artifacts expected by the frozen V4 visual branch.
    """
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess(self, aligned_rgb):
        """
        aligned_rgb: 112x112 RGB numpy array (from align_face)
        Returns: 1x3x224x224 normalized float tensor
        """
        # 1. Simulate dataset save: cv2.imwrite converts to BGR and saves as .jpg
        aligned_bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode('.jpg', aligned_bgr)
        if not success:
            raise ValueError("Failed to encode JPEG for parity adapter")
            
        # 2. Simulate dataset load: Image.open(path).convert('RGB')
        image_pil = Image.open(io.BytesIO(encoded.tobytes())).convert('RGB')
        
        # 3. Apply standard V4 transforms
        face_tensor = self.transform(image_pil).unsqueeze(0)
        
        return face_tensor
