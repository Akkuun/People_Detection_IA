import cv2
import numpy as np
from .EnhanceImageQuality import EnhanceImageQuality

resolution = (125, 125)

class CaptureFace:
    def __init__(self):
        # Détecteurs de visage multiples pour améliorer la détection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Pipeline d'amélioration
        self.enhancer = EnhanceImageQuality()
        
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
    
    def classify_orientation(self, face_img):
        """Classifier si c'est face ou profil"""
        if face_img is None:
            return "unknown"
            
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Détecter les yeux pour déterminer l'orientation
        eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 3, minSize=(10, 10))
        
        if len(eyes) >= 2:
            return "face"  # Deux yeux visibles = face
        elif len(eyes) == 1:
            return "profile"  # Un œil visible = profil
        else:
            # Fallback: analyser la symétrie de l'image
            left_half = gray[:, :gray.shape[1]//2]
            right_half = cv2.flip(gray[:, gray.shape[1]//2:], 1)
            
            # Redimensionner pour comparer
            if left_half.shape != right_half.shape:
                min_width = min(left_half.shape[1], right_half.shape[1])
                left_half = left_half[:, :min_width]
                right_half = right_half[:, :min_width]
            
            # Calculer la différence
            diff = cv2.absdiff(left_half, right_half)
            symmetry_score = np.mean(diff)
            
            return "face" if symmetry_score < 30 else "profile"
    
    def create_mugshot(self, face_img, size=resolution, enhance="minimal"):
        """Créer un mugshot avec options d'amélioration
        
        Args:
            enhance: "none", "minimal", "normal", True (=normal)
        """
        if face_img is None:
            return None, "unknown"
            
        # Redimensionner le visage à la taille demandée
        face_resized = cv2.resize(face_img, size)
        
        # Classifier l'orientation
        orientation = self.classify_orientation(face_resized)
        
        # Améliorer la qualité selon l'option choisie
        if enhance == "none" or enhance == False:
            return face_resized, orientation
        elif enhance == "minimal":
            face_enhanced = self.enhancer.enhance_face_minimal(face_resized)
            return face_enhanced, orientation
        elif enhance == "normal" or enhance == True:
            face_enhanced = self.enhancer.enhance_face(face_resized)
            return face_enhanced, orientation
        else:
            # Par défaut, pas d'amélioration
            return face_resized, orientation