import cv2
import numpy as np

class EnhanceImageQuality:
    def __init__(self):
        pass
    
    def enhance_face(self, face_img):
        """Améliorer la qualité d'une image de visage 125x125 - version douce"""
        if face_img is None:
            return None
            
        enhanced = face_img.copy()
        
        # 1. Débruitage léger seulement
        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)  # Plus doux
        
        # 2. Amélioration du contraste très légère (CLAHE)
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8)).apply(lab[:,:,0])  # Plus doux
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 3. Netteté très légère
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])  # Plus doux
        enhanced = cv2.filter2D(enhanced, -1, kernel * 0.05)  # Moins intense
        
        # 4. PAS d'égalisation d'histogramme (trop agressive pour les petites images)
        
        return enhanced
    
    def enhance_face_minimal(self, face_img):
        """Version minimale : juste un léger débruitage"""
        if face_img is None:
            return None
            
        # Juste un débruitage très léger
        enhanced = cv2.bilateralFilter(face_img, 3, 25, 25)
        return enhanced