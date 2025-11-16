import cv2
import numpy as np

class EnhanceImageQuality:
    def __init__(self):
        pass
    
    def enhance_face(self, face_img):
        if face_img is None:
            return None
            
        # Juste un débruitage très léger
        enhanced = cv2.bilateralFilter(face_img, 3, 25, 25)
        return enhanced
        