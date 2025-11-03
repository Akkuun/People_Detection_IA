from ultralytics import YOLO
import cv2
import numpy as np

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

        label = f"{label_name} {float(conf):.2f}"

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