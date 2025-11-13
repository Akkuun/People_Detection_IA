import cv2
import numpy as np
from .ImageEnhancer import ImageEnhancer

class MugshotPipeline:
    def __init__(self):
        self.enhancer = ImageEnhancer()
        
    def classify_orientation(self, face_img):
        """Classifier si c'est face ou profil"""
        if face_img is None:
            return "unknown"
            
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Détecter les yeux pour déterminer l'orientation
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)
        
        if len(eyes) >= 2:
            return "face"  # Deux yeux visibles = face
        elif len(eyes) == 1:
            return "profile"  # Un œil visible = profil
        else:
            return "unknown"
    
    def process_mugshot(self, face_img):
        """Pipeline complet de traitement"""
        if face_img is None:
            return None, "unknown"
            
        processed = face_img.copy()
        orientation = self.classify_orientation(processed)
        
        # 1. Amélioration qualité
        
        processed = self.enhancer.enhance_face(processed)
        
            
        return processed, orientation