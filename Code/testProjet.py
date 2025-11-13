from ultralytics import YOLO
import cv2
import numpy as np
import sys
import logging

from Mugshot.CaptureFace import CaptureFace

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

frame_count = 0
mugshot_generator = CaptureFace()

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
            print(f"📸 Screenshot sauvegardé: detection_screenshot_{frame_count}.jpg")
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
                        for i, (box, cls) in enumerate(zip(xyxy, clss)):
                            if int(cls) == 0:  # Personne détectée (classe 0 dans YOLO)
                                x1, y1, x2, y2 = map(int, box)
                                bbox = (x1, y1, x2-x1, y2-y1)
                                
                                face = mugshot_generator.extract_face_from_detection(frame, bbox)
                                if face is not None:
                                    mugshot = mugshot_generator.create_mugshot(face)
                                    if mugshot is not None:
                                        cv2.imwrite(f'SavedImages/mugshot_manual_{frame_count}_person{i}.jpg', mugshot)
                                        mugshot_count += 1
                        
                        if mugshot_count > 0:
                            print(f"👤 {mugshot_count} mugshot(s) généré(s)")
                        else:
                            print("❌ Aucune personne détectée pour générer des mugshots")
                    except Exception as e:
                        print(f"Erreur génération mugshot: {e}")
                else:
                    print("❌ Aucune détection disponible")
            else:
                print("❌ Aucun résultat de détection disponible")



finally:
    cap.release()
    cv2.destroyAllWindows()