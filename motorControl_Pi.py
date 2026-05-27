import serial
import time
import ydlidar
from lidarHelpers import init_lidar, get_front_distance

ARDUINO_PORT = "/dev/arduino"

arduino = None  # lazy — not connected until connect() is called

def connect():
    """Call this explicitly during boot, not on import."""
    global arduino
    ser = serial.Serial(ARDUINO_PORT, 9600, timeout=1)
    # Read until READY instead of sleeping a fixed 2 seconds
    deadline = time.time() + 5.0
    while time.time() < deadline:
        line = ser.readline().decode(errors="ignore").strip()
        if line == "READY":
            print(f"[arduino] Connected — got READY")
            break
        if line:
            print(f"[arduino] boot: {line}")
    arduino = ser
    return ser

def wait_for(response, timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        line = arduino.readline().decode().strip()
        if line:
            print(f"Arduino: {line}")
        if line == response:
            return True
    return False

def move(direction, speed, seconds):
    cmd = f"MOVE {direction} {speed} {seconds}\n"
    arduino.write(cmd.encode('utf-8'))
    print(f"Sent: {cmd.strip()}")
    wait_for("DONE", timeout=seconds + 10)

def turn(side, seconds):
    cmd = f"TURN {side} {seconds}\n"
    arduino.write(cmd.encode('utf-8'))
    print(f"Sent: {cmd.strip()}")
    wait_for("DONE", timeout=seconds + 10)

def stop():
    arduino.write(b"STOP\n")
    print("Sent: STOP")
    wait_for("STOPPED", timeout=3)

def go():
    arduino.write(b"GO\n")
    print("Sent: GO")
    wait_for("UNLOCKED", timeout=3)

def moveUntilThreshold(direction, threshold_cm, laser, speed=200):
    cmd = f"MOVE {direction} {speed} 9999\n"
    arduino.write(cmd.encode('utf-8'))
    print(f"Sent: {cmd.strip()}")
    wait_for("MOVING", timeout=3)

    scan = ydlidar.LaserScan()
    while True:
        if laser.doProcessSimple(scan):
            dist = get_front_distance(scan)
            if dist is not None:
                print(f"Forward distance: {dist:.1f}cm")
                if dist <= threshold_cm:
                    stop()
                    go()
                    print(f"Obstacle at {dist:.1f}cm — stopped.")
                    return dist, scan
        time.sleep(0.05)

if __name__ == "__main__":
    connect()
    laser = init_lidar()
    try:
        moveUntilThreshold('F', 15, laser)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        stop()
        laser.turnOff()
        laser.disconnecting()
        arduino.close()