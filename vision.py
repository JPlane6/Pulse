from ultralytics import YOLO
import cv2

def detect_objects(image_path):
    model = YOLO("yolov8n.pt")  # nano model, fastest
    img = cv2.imread(image_path)
    results = model.predict(img)
    detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]
    return detected_classes