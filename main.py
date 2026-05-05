import time
import math
import threading
import cv2
import ydlidar
from RPLCD.i2c import CharLCD
import lcd_hello
import motorControl_Pi as motors
import lidarHelpers
from modules import vision, tts

# --- Constants ---
LIDAR_PORT = "/dev/ttyUSB0"
CAMERA_INDEX = 0
DISPLAY_WIDTH = 960
SPEAK_COOLDOWN = 3.0


# --- Vision Thread ---
def vision_loop():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[vision] ERROR: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("[vision] Camera started — press Q to quit")

    previous_labels = set()
    last_spoken_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[vision] WARNING: Failed to grab frame, retrying...")
                time.sleep(0.05)
                continue

            annotated_frame, current_labels = vision.detect(frame)
            current_set = set(current_labels)

            now = time.time()
            if current_set != previous_labels and (now - last_spoken_time) >= SPEAK_COOLDOWN:
                if current_set:
                    tts.speak(", ".join(sorted(current_set)))
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
                print("[vision] Quit signal received.")
                break

    except KeyboardInterrupt:
        print("[vision] Interrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[vision] Cleaned up.")


# --- Main (LiDAR + Motors + LCD) ---
def main():
    # LCD init
    try:
        lcd = CharLCD("PCF8574", 0x27, cols=20, rows=4)
    except Exception:
        print("[main] ERROR: Could not connect to LCD.")
        return

    lcd.clear()
    lcd_hello.hello()

    # LiDAR init
    laser = ydlidar.CYdLidar()
    laser.setlidaropt(ydlidar.LidarPropSerialPort, LIDAR_PORT)
    laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 128000)
    laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
    laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
    laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)
    laser.setlidaropt(ydlidar.LidarPropSampleRate, 5)
    laser.setlidaropt(ydlidar.LidarPropScanFrequency, 12.0)

    if not laser.initialize():
        lcd.clear()
        lcd.write_string("LIDAR INIT ERROR".ljust(20))
        lcd.cursor_pos = (1, 0)
        lcd.write_string(LIDAR_PORT.ljust(20)[:20])
        return

    if not laser.turnOn():
        lcd.clear()
        lcd.write_string("LIDAR MOTOR ERROR".ljust(20))
        return

    scan = ydlidar.LaserScan()

    # Start vision on a background thread
    vision_thread = threading.Thread(target=vision_loop, daemon=True)
    vision_thread.start()

    # LiDAR + motor loop on main thread
    try:
        while True:
            if lidarHelpers.is_obstacle_ahead(scan, 30):
                motors.stop()
            else:
                motors.go()
                motors.moveUntilThreshold("FORWARD", 200, 30, laser)

            if laser.doProcessSimple(scan):
                left_cm, right_cm = lidarHelpers.get_left_right_distances_cm(scan)
                lcd.cursor_pos = (0, 0)
                lcd.write_string("LIDAR LEFT/RIGHT".ljust(20))
                lcd.cursor_pos = (1, 0)
                lcd.write_string(f"LEFT : {left_cm:6.1f}cm".ljust(20))
                lcd.cursor_pos = (2, 0)
                lcd.write_string(f"RIGHT: {right_cm:6.1f}cm".ljust(20))
                lcd.cursor_pos = (3, 0)
                lcd.write_string("RUNNING".ljust(20))

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("[main] Interrupted.")
        motors.stop()
    finally:
        laser.turnOff()
        laser.disconnecting()
        lcd.clear()
        lcd.write_string("SYSTEM IDLE".ljust(20))
        motors.go()


if __name__ == "__main__":
    main()