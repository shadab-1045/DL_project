import os
import insightface
import cv2

class NativeInSwapper:
    def __init__(self, model_path="models/inswapper_128.onnx"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Native InSwapper model not found at {model_path}")
        
        # Load the model once
        self.swapper = insightface.model_zoo.get_model(model_path, download=False, download_zip=False)
        self.source_face = None

    def set_source_face(self, source_face):
        """
        Sets the source identity (e.g. Alice) to be swapped onto the live webcam feed.
        source_face should be an insightface Face object.
        """
        self.source_face = source_face

    def process_frame(self, target_frame_bgr, target_faces):
        """
        Swaps the source_face onto the target_faces in the target_frame.
        target_faces should be a list of insightface Face objects detected in the target_frame.
        """
        if self.source_face is None:
            return target_frame_bgr
        
        if not target_faces:
            return target_frame_bgr
            
        result_frame = target_frame_bgr.copy()
        
        # Swap onto all detected faces in the target frame
        for face in target_faces:
            result_frame = self.swapper.get(result_frame, face, self.source_face, paste_back=True)
            
        return result_frame
