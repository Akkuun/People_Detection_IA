import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Force X11 instead of Wayland

from ultralytics import YOLO
import cv2
import numpy as np
import sys
import logging
import time

from Utility.CaptureFace import CaptureFace
from Utility.MugshotPipeline import MugshotPipeline
from inference_cyclegan import CycleGANInference
from inference_vaegan import VAEGANInference

# Supprimer les logs verbeux de YOLO et ultralytics
logging.getLogger('ultralytics').setLevel(logging.WARNING)

# Chargement modèle YOLO
try:
    model = YOLO("yolov8n.pt", verbose=False)
except Exception as e:
    sys.exit(1)

# Test webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Impossible d'ouvrir la caméra.")
    sys.exit(1)

print("wEBCAM OK")

# Test lecture flux
ret, test_frame = cap.read()
if not ret:
    print("Impossible de lire l'image de la caméra.")
    cap.release()
    sys.exit(1)

cv2.namedWindow("Détection webcam", cv2.WINDOW_NORMAL)

# Fonction pour dessiner les boîtes sur une image OpenCV
def draw_boxes_opencv(frame, results):
    if results is None or len(results) == 0: #si résultats vides, on affiche rien du tout
        return

    res = results[0]
    
    boxes = getattr(res, "boxes", None)
    if boxes is None: # on recupère les boîtes, si aucune boîte, on ne fait rien
        return

    # Extraction des données des boîtes
    try:
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)
    except Exception:
        # On essaie une autre méthode d'extraction
        try:
            xyxy = np.array([b.xyxy for b in boxes])
            confs = np.array([b.conf for b in boxes])
            clss = np.array([b.cls for b in boxes])
        except Exception:
            return

    # Dessiner chaque boite détectées
    for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
        try:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        except Exception:
            # Si conversion échoue, on skip cette boîte
            continue

        label_name = None
        if hasattr(model, "names") and model.names is not None:
            # Récupérer le nom de la classe
            try:
                label_name = model.names[int(cls)]
            except Exception:
                label_name = str(int(cls))
        else:
            label_name = str(int(cls))

        label = f"{label_name} {float(conf):.2f}"

        # Dessiner la boîte et le label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (16, 255, 16), 2)
        # Fond pour le texte pour la lisibilité
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - 6), (x1 + text_w, y1), (16, 255, 16), -1)
        cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


# MAIN LOOP
print("   - Appuyez sur 'q' ou 'ESC' pour quitter")
print("   - Appuyez sur 's' pour capturer une image")
print("   - Appuyez sur 'm' pour générer les mugshots")
print("   - Appuyez sur 'c' pour détecter et classifier les visages")
print("   - Appuyez sur 'y' pour styliser l'image avec le style du dataset")
print("   - Appuyez sur '1' pour utiliser CycleGAN (256x256)")
print("   - Appuyez sur '2' pour utiliser VAE-GAN (128x128)")

frame_count = 0
mugshot_generator = CaptureFace()
mugshot_pipeline = MugshotPipeline()
cyclegan_inference = CycleGANInference(image_size=256)
vaegan_inference = VAEGANInference(image_size=128)

# Modèle par défaut pour le style transfer
current_style_model = 'cyclegan'  # ou 'vaegan'

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Impossible de lire l'image de la caméra")
            break

        frame_count += 1

        # Exécution du modèle YOLO
        try:
            results = model(frame, verbose=False)
        except Exception as e:
            print(f"Erreur lors de l'inférence: {e}")
            annotated = frame
        else:
            # Dessiner les boîtes sur l'image
            annotated = frame.copy()
            try:
                draw_boxes_opencv(annotated, results)
            except Exception as e:
                print(f"Erreur lors du dessin des boîtes: {e}")
                try:
                    annotated = results[0].plot()
                except Exception:
                    annotated = frame

        cv2.imshow("Détection webcam", annotated)

        # On regarde si l'utilisateur veut quitter ou sauvegarder une image
        key = cv2.waitKey(1) & 0xFF
        # si touche == Q ou touche == ESC, on quitte
        if key == 27 or key == ord('q'):
            print("🛑 Arrêt demandé par l'utilisateur")
            break
        # si touche == S, on sauvegarde une image en JPG
        elif key == ord('s'): 
            cv2.imwrite(f'SavedImages/detection_screenshot_{frame_count}.jpg', annotated)
            print(f" Screenshot sauvegardé: detection_screenshot_{frame_count}.jpg")
        # si touche == M, on génère les mugshots
        elif key == ord('m'):
            if 'results' in locals() and results and len(results) > 0:
                res = results[0]
                boxes = getattr(res, "boxes", None)
                
                if boxes is not None:
                    try:
                        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
                        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)
                        
                        mugshot_count = 0
                        timestamp = int(time.time())
                        
                        for i, (box, cls) in enumerate(zip(xyxy, clss)):
                            if int(cls) == 0:  # Personne détectée (classe 0 dans YOLO)
                                x1, y1, x2, y2 = map(int, box)
                                bbox = (x1, y1, x2-x1, y2-y1)
                                
                                # Extraire le visage
                                face = mugshot_generator.extract_face_from_detection(frame, bbox)
                                if face is not None:
                                    # Utiliser le nouveau pipeline complet
                                    processed_face, orientation = mugshot_pipeline.process_mugshot(face)
                                    
                                    if processed_face is not None:
                                        filename = f'SavedImages/mugshot_{orientation}_{timestamp}_person{i}.jpg'
                                        cv2.imwrite(filename, processed_face)
                                        mugshot_count += 1
                                        print(f"  📸 {orientation}: {filename}")
                        
                        if mugshot_count > 0:
                            print(f"{mugshot_count} mugshot(s) généré(s) avec le pipeline complet")
                        else:
                            print(" Aucune personne détectée pour générer des mugshots")
                    except Exception as e:
                        print(f"Erreur génération mugshot: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(" Aucune détection disponible")
            else:
                print(" Aucun résultat de détection disponible")
        # si touche == C, on détecte et classifie les visages
        elif key == ord('c'):
            if 'results' in locals() and results and len(results) > 0:
                res = results[0]
                boxes = getattr(res, "boxes", None)
                
                if boxes is not None:
                    try:
                        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
                        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)
                        
                        face_count = 0
                        timestamp = int(time.time())
                        
                        for i, (box, cls) in enumerate(zip(xyxy, clss)):
                            if int(cls) == 0:  # Personne détectée (classe 0 dans YOLO)
                                x1, y1, x2, y2 = map(int, box)
                                bbox = (x1, y1, x2-x1, y2-y1)
                                
                                # Extraire le visage
                                face = mugshot_generator.extract_face_from_detection(frame, bbox)
                                if face is not None:
                                    # Classifier l'orientation seulement
                                    orientation = mugshot_generator.classify_orientation(face)
                                    
                                    # Sauvegarder l'image avec le nom de l'orientation détectée
                                    filename = f'SavedImages/{orientation}_{timestamp}_person{i}.jpg'
                                    cv2.imwrite(filename, face)
                                    face_count += 1
                                    print(f"  🔍 Détecté {orientation}: {filename}")
                        
                        if face_count > 0:
                            print(f"{face_count} visage(s) détecté(s) et classifié(s)")
                        else:
                            print(" Aucune personne détectée pour la classification de visage")
                    except Exception as e:
                        print(f"Erreur détection visage: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(" Aucune détection disponible")
            else:
                print(" Aucun résultat de détection disponible")
        # si touche == 1, on bascule vers CycleGAN
        elif key == ord('1'):
            current_style_model = 'cyclegan'
            print("🔄 Passage au modèle CycleGAN (256x256)")
        # si touche == 2, on bascule vers VAE-GAN
        elif key == ord('2'):
            current_style_model = 'vaegan'
            print("🔄 Passage au modèle VAE-GAN (128x128)")
        # si touche == Y, on applique le style transfer
        elif key == ord('y'):
            if current_style_model == 'cyclegan':
                style_model = cyclegan_inference
                model_name = "CycleGAN"
            else:
                style_model = vaegan_inference
                model_name = "VAE-GAN"
            
            if style_model.G is None:
                print(f"❌ Modèle {model_name} non chargé!")
                print(f"   Solution: Entraînez le modèle avec: python3 train.py")
                print(f"   Puis redémarrez l'app")
            elif 'results' in locals() and results and len(results) > 0:
                res = results[0]
                boxes = getattr(res, "boxes", None)
                
                if boxes is not None:
                    try:
                        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
                        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)
                        
                        style_count = 0
                        timestamp = int(time.time())
                        
                        for i, (box, cls) in enumerate(zip(xyxy, clss)):
                            if int(cls) == 0:  # Personne détectée
                                x1, y1, x2, y2 = map(int, box)
                                bbox = (x1, y1, x2-x1, y2-y1)
                                
                                # Extraire le visage
                                face = mugshot_generator.extract_face_from_detection(frame, bbox)
                                if face is not None:
                                    # Appliquer le style transfer (CycleGAN ou VAE-GAN)
                                    styled_face = style_model.apply_style(face)
                                    
                                    if styled_face is not None:
                                        filename = f'SavedImages/styled_{current_style_model}_{timestamp}_person{i}.jpg'
                                        cv2.imwrite(filename, styled_face)
                                        style_count += 1
                                        print(f"  🎨 [{model_name}] Stylisé: {filename}")
                        
                        if style_count > 0:
                            print(f"{style_count} visage(s) stylisé(s) avec {model_name}")
                        else:
                            print(" Aucune personne détectée pour styliser")
                    except Exception as e:
                        print(f"Erreur style transfer {model_name}: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(" Aucune détection disponible")
            else:
                print(" Aucun résultat de détection disponible")


finally:
    cap.release()
    cv2.destroyAllWindows()