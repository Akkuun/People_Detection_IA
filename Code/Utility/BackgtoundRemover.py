import cv2
import numpy as np

class BackgroundRemover:
    def __init__(self):
        pass
    
    def remove_background(self, face_img):
        """Remplacer le fond par du gris uniforme"""
        if face_img is None:
            return None
            
        # 1. Créer un masque simple basé sur la détection de contours
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # 2. Seuillage adaptatif pour séparer visage/fond
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # 3. Morphologie pour nettoyer le masque
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 4. Trouver le plus grand contour (visage)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [largest_contour], 255)
            
            # 5. Appliquer le masque avec fond gris
            result = face_img.copy()
            background = np.full_like(face_img, (128, 128, 128))  # Gris uniforme
            mask_3d = cv2.merge([mask, mask, mask]) / 255.0
            result = (face_img * mask_3d + background * (1 - mask_3d)).astype(np.uint8)
            
            return result
        
        return face_img