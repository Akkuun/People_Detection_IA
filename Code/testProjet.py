from ultralytics import YOLO
import cv2
import numpy as np
from deepface import DeepFace
import traceback

# Load model once
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Détection webcam", cv2.WINDOW_NORMAL)


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

        # Attempt gender detection for persons using DeepFace
        gender = ""
        if label_name == "person":
            # lightweight caching + rate limiting to avoid analyzing every frame
            # cache key: quantized center position
            try:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
            except Exception:
                cx, cy = 0, 0

            # Access / initialize frame counter & cache on function attribute to avoid globals
            if not hasattr(draw_boxes_opencv, "frame_id"):
                draw_boxes_opencv.frame_id = 0
            if not hasattr(draw_boxes_opencv, "gender_cache"):
                draw_boxes_opencv.gender_cache = {}

            FRAME_SKIP = 10  # analyze every N frames
            CACHE_TTL = 30   # keep cached result for N frames
            GRID = 50        # quantization size for spatial key

            key = (cx // GRID, cy // GRID)
            cached = draw_boxes_opencv.gender_cache.get(key)
            use_cached = False
            if cached is not None:
                cached_gender, cached_frame = cached
                if draw_boxes_opencv.frame_id - cached_frame <= CACHE_TTL:
                    gender = cached_gender
                    use_cached = True

            if not use_cached and (draw_boxes_opencv.frame_id % FRAME_SKIP == 0):
                # Take a larger top portion of the detected box as an approximate face region
                # and add a small horizontal padding so faces near the sides aren't cut off.
                face_height = int((y2 - y1) * 0.6)
                pad = int((x2 - x1) * 0.1)
                fy1 = max(0, y1)
                fy2 = min(frame.shape[0], y1 + max(1, face_height))
                fx1 = max(0, x1 - pad)
                fx2 = min(frame.shape[1], x2 + pad)
                face_region = frame[fy1:fy2, fx1:fx2]

                # Ensure region is valid and large enough
                h = fy2 - fy1
                w = fx2 - fx1
                MIN_FACE = 32
                if h >= MIN_FACE and w >= MIN_FACE:
                    try:
                        # DeepFace expects RGB images
                        face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                        face_small = cv2.resize(face_rgb, (224, 224))
                        # Force a robust detector backend (mtcnn) and log result for debugging
                        result = DeepFace.analyze(
                            face_small,
                            actions=['gender'],
                            enforce_detection=False,
                            detector_backend='mtcnn',
                            silent=True,
                        )
                        print(f"[DeepFace] frame={draw_boxes_opencv.frame_id} key={key} size={w}x{h}")
                        print(f"[DeepFace] raw result: {result}")
                        if isinstance(result, list) and len(result) > 0:
                            entry = result[0]
                        else:
                            entry = result
                        gender = entry.get('dominant_gender') or entry.get('gender') or "unknown"
                        # normalize common values
                        if isinstance(gender, str):
                            g = gender.lower()
                            if 'male' in g or 'man' in g:
                                gender = 'Male'
                            elif 'female' in g or 'woman' in g:
                                gender = 'Female'
                            else:
                                gender = gender.capitalize()
                        else:
                            gender = str(gender)
                        # store in cache
                        draw_boxes_opencv.gender_cache[key] = (gender, draw_boxes_opencv.frame_id)
                    except Exception:
                        # print the traceback once to help debugging
                        traceback.print_exc()
                        gender = "unknown"
                else:
                    gender = "unknown"

            # increment frame counter
            draw_boxes_opencv.frame_id += 1

        label = f"{label_name} {float(conf):.2f}"
        if label_name == "person" and gender:
            label += f" ({gender})"

        # Draw rectangle and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (16, 255, 16), 2)
        # Background for text for readability
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - 6), (x1 + text_w, y1), (16, 255, 16), -1)
        cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run model inference (do not call any show/plot helpers that may open windows)
        results = model(frame)

        # Draw boxes on a copy of the frame
        annotated = frame.copy()
        try:
            draw_boxes_opencv(annotated, results)
        except Exception:
            # as a last resort fallback to the model's plot if available
            try:
                annotated = results[0].plot()
            except Exception:
                # If even that fails, just show the original frame
                annotated = frame

        cv2.imshow("Détection webcam", annotated)

        # ESC pour quitter
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()