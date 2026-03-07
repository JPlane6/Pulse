import cv2
from ultralytics import YOLO
import torch

# Load model once at import time
model = YOLO("yolo26x.pt")

# Use MPS (Apple Silicon GPU) if available, else CPU
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[vision] Running on: {DEVICE}")

FRAME_SKIP = 2          # Run YOLO every N frames
RESIZE_WIDTH = 640      # Resize input before YOLO (balance speed vs accuracy)

_frame_count = 0
_last_results = []      # Cache last detections for skipped frames


def detect(frame):
    global _frame_count, _last_results
    _frame_count += 1

    h, w = frame.shape[:2]
    scale = RESIZE_WIDTH / w
    resized = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))

    if _frame_count % FRAME_SKIP == 0:
        results = model(resized, device=DEVICE, verbose=False)[0]
        _last_results = results
    else:
        results = _last_results

    if not results:
        return frame, []

    labels = []
    annotated = frame.copy()

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = (
            int(x1 / scale), int(y1 / scale),
            int(x2 / scale), int(y2 / scale)
        )
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if conf < 0.65:
            continue

        labels.append(label)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label_text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y2), (x1 + tw, y2 + th + 6), (0, 255, 0), -1)
        cv2.putText(
            annotated, label_text,
            (x1, y2 + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA
        )

    return annotated, list(set(labels))