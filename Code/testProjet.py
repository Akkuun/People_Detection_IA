from ultralytics import YOLO
import cv2
import numpy as np
import sys

print("🚀 Démarrage de l'application de détection...")

# Load model once
try:
    print("📥 Chargement du modèle YOLO...")
    model = YOLO("yolov8n.pt")
    print("✅ Modèle YOLO chargé avec succès")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle: {e}")
    sys.exit(1)

# Open webcam
print("📹 Tentative d'ouverture de la caméra...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Erreur: Impossible d'ouvrir la caméra!")
    print("💡 Vérifications à faire:")
    print("   - La caméra est-elle connectée?")
    print("   - Une autre application utilise-t-elle la caméra?")
    print("   - Permissions d'accès à la caméra?")
    sys.exit(1)

print("✅ Caméra ouverte avec succès")

# Test read frame
ret, test_frame = cap.read()
if not ret:
    print("❌ Erreur: Impossible de lire une image depuis la caméra!")
    cap.release()
    sys.exit(1)

print(f"✅ Image test lue: {test_frame.shape}")

cv2.namedWindow("Détection webcam", cv2.WINDOW_NORMAL)
print("✅ Fenêtre créée")


def draw_boxes_opencv(frame, results):
    """Draw boxes and labels from ultralytics Results onto an OpenCV frame.
    This avoids calling any plotting/display functions from ultralytics that
    may spawn extra windows.
    """
    if results is None or len(results) == 0:
        return

    res = results[0]
    # results[0].boxes may be an object with attributes xyxy, conf, cls
    boxes = getattr(res, "boxes", None)
    if boxes is None:
        return

    # Convert tensors to numpy if necessary
    try:
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)
    except Exception:
        # Fallback: try to iterate boxes directly
        try:
            xyxy = np.array([b.xyxy for b in boxes])
            confs = np.array([b.conf for b in boxes])
            clss = np.array([b.cls for b in boxes])
        except Exception:
            return

    # Draw each detection
    for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
        try:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        except Exception:
            # If conversion fails, skip this box
            continue

        label_name = None
        if hasattr(model, "names") and model.names is not None:
            # model.names is usually a dict or list
            try:
                label_name = model.names[int(cls)]
            except Exception:
                label_name = str(int(cls))
        else:
            label_name = str(int(cls))

        label = f"{label_name} {float(conf):.2f}"

        # Draw rectangle and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (16, 255, 16), 2)
        # Background for text for readability
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - 6), (x1 + text_w, y1), (16, 255, 16), -1)
        cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


print("🎥 Démarrage de la boucle de détection...")
print("💡 Instructions:")
print("   - Appuyez sur 'q' ou 'ESC' pour quitter")
print("   - Appuyez sur 's' pour capturer une image")

frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Erreur: Impossible de lire l'image de la caméra")
            break

        frame_count += 1
        if frame_count % 30 == 0:  # Print every 30 frames
            print(f"📊 Frame #{frame_count} traité")

        # Run model inference (do not call any show/plot helpers that may open windows)
        try:
            results = model(frame)
        except Exception as e:
            print(f"❌ Erreur lors de l'inférence: {e}")
            annotated = frame
        else:
            # Draw boxes on a copy of the frame
            annotated = frame.copy()
            try:
                draw_boxes_opencv(annotated, results)
            except Exception as e:
                print(f"⚠️ Erreur lors du dessin des boîtes: {e}")
                # as a last resort fallback to the model's plot if available
                try:
                    annotated = results[0].plot()
                except Exception:
                    # If even that fails, just show the original frame
                    annotated = frame

        cv2.imshow("Détection webcam", annotated)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or 'q' to quit
            print("🛑 Arrêt demandé par l'utilisateur")
            break
        elif key == ord('s'):  # 's' to save screenshot
            cv2.imwrite(f'detection_screenshot_{frame_count}.jpg', annotated)
            print(f"📸 Screenshot sauvegardé: detection_screenshot_{frame_count}.jpg")

except KeyboardInterrupt:
    print("🛑 Interruption clavier (Ctrl+C)")
except Exception as e:
    print(f"❌ Erreur inattendue: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("🧹 Nettoyage des ressources...")
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Application fermée proprement")