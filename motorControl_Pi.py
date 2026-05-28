import serial
import time
import ydlidar
from lidarHelpers import init_lidar, get_front_distance

ARDUINO_PORT = "/dev/arduino"
arduino = None  # lazy — not connected until connect() is called


def connect():
    global arduino
    ser = serial.Serial(ARDUINO_PORT, 9600, timeout=1)
    deadline = time.time() + 5.0
    while time.time() < deadline:
        line = ser.readline().decode(errors="ignore").strip()
        if line == "READY":
            print("[arduino] Connected — READY")
            break
        if line:
            print(f"[arduino] boot: {line}")
    arduino = ser
    return ser


def wait_for(response, timeout=5):
    start = time.time()
    while time.time() - start < timeout:
        line = arduino.readline().decode().strip()
        if line: print(f"Arduino: {line}")
        if line == response: return True
    return False


def move(direction, speed, seconds):
    arduino.write(f"MOVE {direction} {speed} {seconds}\n".encode())
    wait_for("DONE", timeout=seconds + 5)


def turn(side, seconds):
    arduino.write(f"TURN {side} {seconds}\n".encode())
    wait_for("DONE", timeout=seconds + 5)


def stop():
    arduino.write(b"STOP\n")
    wait_for("STOPPED", timeout=2)


def go():
    arduino.write(b"GO\n")
    wait_for("UNLOCKED", timeout=2)


def moveUntilThreshold(direction, threshold_cm, laser, speed=200):
    arduino.write(f"MOVE {direction} {speed} 9999\n".encode())
    wait_for("MOVING", timeout=1)
    scan = ydlidar.LaserScan()
    while True:
        if laser.doProcessSimple(scan):
            dist = get_front_distance(scan)
            if dist is not None:
                print(f"[move] {dist:.1f}cm", end='\r')
                if dist <= threshold_cm:
                    stop(); go()
                    print(f"\n[move] Stopped at {dist:.1f}cm")
                    return dist, scan


if __name__ == "__main__":
    connect()
    laser = init_lidar()
    try:
        moveUntilThreshold('F', 15, laser)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        stop()
        laser.turnOff(); laser.disconnecting()
        arduino.close()