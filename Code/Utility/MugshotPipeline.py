import cv2
import numpy as np
from .EnhanceImageQuality import EnhanceImageQuality
from .FaceGenerator import FaceGenerator

class MugshotPipeline:
    def __init__(self):
        self.enhancer = EnhanceImageQuality()
        self.generator = FaceGenerator()
        
    def classify_orientation(self, face_img):
        """Classifier si c'est face ou profil avec plus de précision"""
        if face_img is None:
            return "unknown"
            
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Score pour chaque orientation
        face_score = 0
        profile_score = 0
        
        # 1. Détection des caractéristiques faciales
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Détection avec plusieurs paramètres pour plus de robustesse
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
        profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(5, 5))
        
        # Profil inversé (flip horizontal pour détecter profil droit)
        gray_flipped = cv2.flip(gray, 1)
        profiles_flipped = profile_cascade.detectMultiScale(gray_flipped, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
        
        # 2. Score basé sur les détections
        if len(faces) > 0:
            face_score += 3
        if len(profiles) > 0 or len(profiles_flipped) > 0:
            profile_score += 3
            
        # 3. Analyse des yeux
        if len(eyes) >= 2:
            # Vérifier si les yeux sont alignés horizontalement (face)
            eye_centers = [(x + w//2, y + h//2) for x, y, w, h in eyes]
            if len(eye_centers) >= 2:
                # Prendre les 2 yeux les plus probables (plus grands)
                eye_areas = [(w*h, i) for i, (x, y, w, h) in enumerate(eyes)]
                eye_areas.sort(reverse=True)
                
                if len(eye_areas) >= 2:
                    eye1_idx = eye_areas[0][1]
                    eye2_idx = eye_areas[1][1]
                    y1 = eye_centers[eye1_idx][1]
                    y2 = eye_centers[eye2_idx][1]
                    
                    # Si les yeux sont à peu près à la même hauteur
                    if abs(y1 - y2) < height * 0.1:
                        face_score += 4
                    else:
                        profile_score += 2
                        
            face_score += len(eyes)
        elif len(eyes) == 1:
            profile_score += 2
            
        # 4. Analyse de symétrie
        left_half = gray[:, :width//2]
        right_half = gray[:, width//2:]
        
        if right_half.shape[1] != left_half.shape[1]:
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
        
        right_half_flipped = cv2.flip(right_half, 1)
        
        # Calculer la corrélation entre les deux moitiés
        correlation = cv2.matchTemplate(left_half, right_half_flipped, cv2.TM_CCOEFF_NORMED)[0][0]
        
        if correlation > 0.7:  # Haute symétrie = face
            face_score += 2
        elif correlation < 0.4:  # Faible symétrie = profil
            profile_score += 2
            
        # 5. Analyse de la distribution des contours
        edges = cv2.Canny(gray, 50, 150)
        
        # Diviser en zones pour analyser la distribution
        left_zone = edges[:, :width//3]
        center_zone = edges[:, width//3:2*width//3]
        right_zone = edges[:, 2*width//3:]
        
        left_edges = np.sum(left_zone)
        center_edges = np.sum(center_zone)
        right_edges = np.sum(right_zone)
        
        # Pour un profil, une zone devrait avoir beaucoup plus de contours
        total_edges = left_edges + center_edges + right_edges
        if total_edges > 0:
            left_ratio = left_edges / total_edges
            right_ratio = right_edges / total_edges
            
            # Si une zone latérale domine (>50%), c'est probablement un profil
            if left_ratio > 0.5 or right_ratio > 0.5:
                profile_score += 1
            else:
                face_score += 1
        
        # 6. Décision finale
        print(f"Debug - Face score: {face_score}, Profile score: {profile_score}")
        
        if face_score > profile_score and face_score >= 3:
            return "face"
        elif profile_score > face_score and profile_score >= 3:
            return "profile"
        else:
            # Fallback sur l'analyse simple des yeux
            if len(eyes) >= 2:
                return "face"
            elif len(eyes) == 1 or len(profiles) > 0 or len(profiles_flipped) > 0:
                return "profile"
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
        
        # 2. Génération vue frontale si profil
        if orientation == "profile":
            processed, orientation = self.generator.enhance_frontal_generation(processed, orientation)
        else:
            # 3. Normalisation de la taille pour les vues frontales
            processed = self.generator.normalize_mugshot_size(processed, (200, 240))
            
        return processed, orientation