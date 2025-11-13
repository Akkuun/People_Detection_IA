import cv2
import numpy as np

resolution = (125, 125)

class CaptureFace:
    def __init__(self):
        # Détecteurs de visage multiples pour améliorer la détection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # Paramètres de détection plus flexibles
        self.detection_params = [
            {'scaleFactor': 1.1, 'minNeighbors': 3, 'minSize': (30, 30)},
            {'scaleFactor': 1.05, 'minNeighbors': 2, 'minSize': (20, 20)},
            {'scaleFactor': 1.3, 'minNeighbors': 5, 'minSize': (40, 40)}
        ]
        
    def extract_face_from_detection(self, frame, bbox):
        """Extraire le visage depuis votre bounding box de détection de personne"""
        x, y, w, h = bbox
        person_roi = frame[y:y+h, x:x+w]
        
        # Essayer plusieurs méthodes de détection
        all_faces = []
        
        # 1. Détection frontale avec plusieurs paramètres
        for params in self.detection_params:
            faces = self.face_cascade.detectMultiScale(person_roi, **params)
            all_faces.extend(faces)
        
        # 2. Détection de profil (gauche et droite)
        try:
            # Profil normal
            profile_faces = self.profile_cascade.detectMultiScale(person_roi, 1.1, 3, minSize=(30, 30))
            all_faces.extend(profile_faces)
            
            # Profil inversé (flip horizontal)
            flipped_roi = cv2.flip(person_roi, 1)
            profile_faces_flipped = self.profile_cascade.detectMultiScale(flipped_roi, 1.1, 3, minSize=(30, 30))
            # Ajuster les coordonnées pour le flip
            for (fx, fy, fw, fh) in profile_faces_flipped:
                fx_adjusted = person_roi.shape[1] - fx - fw
                all_faces.append((fx_adjusted, fy, fw, fh))
        except Exception:
            pass  # Si le détecteur de profil n'est pas disponible
        
        # 3. Si aucun visage détecté, prendre la partie supérieure de la personne
        if len(all_faces) == 0:
            # Fallback: prendre le tiers supérieur de la détection de personne
            head_height = int(h * 0.4)  # 40% de la hauteur pour la tête
            head_width = int(w * 0.6)   # 60% de la largeur centrée
            head_x = int(w * 0.2)       # Centrer horizontalement
            
            if head_height > 20 and head_width > 20:  # Vérifier taille minimum
                face = person_roi[0:head_height, head_x:head_x+head_width]
                return face
            return None
        
        # Prendre le plus grand visage détecté parmi tous
        fx, fy, fw, fh = max(all_faces, key=lambda f: f[2]*f[3])
        
        # Agrandir légèrement la zone pour capturer plus de contexte
        margin = int(min(fw, fh) * 0.2)  # 20% de marge
        fx = max(0, fx - margin)
        fy = max(0, fy - margin)
        fw = min(person_roi.shape[1] - fx, fw + 2*margin)
        fh = min(person_roi.shape[0] - fy, fh + 2*margin)
        
        face = person_roi[fy:fy+fh, fx:fx+fw]
        return face
    
    def create_mugshot(self, face_img, size=resolution):
        """Créer un mugshot : juste le visage redimensionné sans bordures"""
        if face_img is None:
            return None
            
        # Redimensionner le visage à la taille demandée
        face_resized = cv2.resize(face_img, size)
        
        return face_resized