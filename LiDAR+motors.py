import serial
import ydlidar
import time
import motorControl_Pi as motors
from PULSE_Lidar_Test1 import get_distance

# --- LiDAR Setup ---
laser = ydlidar.CYdLidar()
laser.setlidaropt(ydlidar.LidarPropSerialPort, "/dev/ttyUSB0")
laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 128000)
laser.setlidaropt(ydlidar.LidarPropLidarType, 18)
laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)

if not laser.initialize():
    print("Failed to initialize LiDAR")
    exit()

if not laser.turnOn():
    print("Failed to start motor")
    exit()

scan = ydlidar.LaserScan()

# --- Main Loop ---
print("Starting... Press Ctrl+C to stop.")

try:
    while True:
        if laser.doProcessSimple(scan):
            distance = get_distance(scan)

            if distance is not None:
                # Obstacle detected — stop and wait
                print(f"Obstacle detected at {distance:.2f} in — stopping")
                motors.stop()

                # Wait until obstacle is removed
                while True:
                    if laser.doProcessSimple(scan):
                        distance = get_distance(scan)
                        if distance is None:
                            print("Obstacle removed — moving forward")
                            break
                    time.sleep(0.05)

            else:
                # No obstacle — keep moving forward
                motors.go()
                motors.move('F', 200, 1)

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    motors.stop()
    arduino.close()
    laser.turnOff()
    laser.disconnecting()