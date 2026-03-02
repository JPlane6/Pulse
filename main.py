# main.py
from modules.vision import detect_objects
from modules.tts import speak

image_path = "data/sample.jpg"  # change to your test image
detected = detect_objects(image_path)

print(f"Detected objects: {detected}")

if detected:
    speak(f"I see a {', '.join(detected)}")