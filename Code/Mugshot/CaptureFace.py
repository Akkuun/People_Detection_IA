import cv2
import numpy as np

class CaptureFace:
    def __init__(self):
        # Détecteur de visage
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def extract_face_from_detection(self, frame, bbox):
        """Extraire le visage depuis votre bounding box de détection de personne"""
        x, y, w, h = bbox
        person_roi = frame[y:y+h, x:x+w]
        
        # Détecter le visage dans la zone de la personne
        faces = self.face_cascade.detectMultiScale(person_roi, 1.1, 4)
        
        if len(faces) > 0:
            # Prendre le plus grand visage détecté
            fx, fy, fw, fh = max(faces, key=lambda f: f[2]*f[3])
            face = person_roi[fy:fy+fh, fx:fx+fw]
            return face
        return None
    
    def create_mugshot(self, face_img, size=(125, 125)):
        """Créer un mugshot : juste le visage redimensionné sans bordures"""
        if face_img is None:
            return None
            
        # Redimensionner le visage à la taille demandée
        face_resized = cv2.resize(face_img, size)
        
        return face_resized