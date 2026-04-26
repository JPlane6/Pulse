import serial
import time
import ydlidar
from lidarHelpers import init_lidar, get_front_distance

ARDUINO_PORT = "/dev/ttyACM0"

def _connect_arduino():
    try:
        ser = serial.Serial(ARDUINO_PORT, 9600, timeout=1)
        time.sleep(2)
        print(f"Arduino: {ser.readline().decode().strip()}")
        return ser
    except serial.SerialException:
        print(f"Error: Could not connect to Arduino on {ARDUINO_PORT}")
        exit()

arduino = _connect_arduino()

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

def moveUntilThreshold(direction, speed = 200, threshold_cm, laser):
    """Move until obstacle within threshold_cm — then stop and return."""
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
                    print(f"Obstacle at {dist:.1f}cm — stopped.")
                    return dist, scan
        time.sleep(0.05)

if __name__ == "__main__":
    laser = init_lidar()
    try:
        # move('F', 200, 1)
        # move('B', 200, 1)
        # turn('L', 1)
        # turn('R', 1)
        moveUntilThreshold('F', 200, 15, laser)  # stop at 10cm
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        stop()
        laser.turnOff()
        laser.disconnecting()
        arduino.close()