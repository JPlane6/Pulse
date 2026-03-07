import cv2
import time
from modules import vision, tts

CAMERA_INDEX = 0
DISPLAY_WIDTH = 960
SPEAK_COOLDOWN = 3.0


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[main] ERROR: Could not open camera. Check CAMERA_INDEX.")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("[main] Starting PulseAI — press Q to quit")

    previous_labels = set()
    last_spoken_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[main] WARNING: Failed to grab frame, retrying...")
                time.sleep(0.05)
                continue

            annotated_frame, current_labels = vision.detect(frame)
            current_set = set(current_labels)

            now = time.time()
            if current_set != previous_labels and (now - last_spoken_time) >= SPEAK_COOLDOWN:
                if current_set:
                    speech = ", ".join(sorted(current_set))
                    tts.speak(speech)
                    last_spoken_time = now
                previous_labels = current_set

            h, w = annotated_frame.shape[:2]
            display_scale = DISPLAY_WIDTH / w
            display_frame = cv2.resize(
                annotated_frame,
                (DISPLAY_WIDTH, int(h * display_scale))
            )

            cv2.putText(
                display_frame, f"FPS: {1.0 / max(time.time() - now, 1e-6):.1f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2, cv2.LINE_AA
            )

            cv2.imshow("PulseAI", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[main] Quit signal received.")
                break

    except KeyboardInterrupt:
        print("[main] Interrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[main] Cleaned up.")


if __name__ == "__main__":
    main()