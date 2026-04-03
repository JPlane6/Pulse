#Prototype 1 - Pulse YDLiDAR
import ydlidar
import time

# Helper function to convert meters to inches
def m_to_in(meters):
    return meters * 39.3701

def get_distance(scan):
    """Returns the closest valid distance in inches from a scan, or None if nothing detected."""
    closest = None
    for point in scan.points:
        dist_inches = m_to_in(point.range)
        if 0.1 < dist_inches <= 5.0:
            if closest is None or dist_inches < closest:
                closest = dist_inches
    return closest

def main():
    laser = ydlidar.CYdLidar()

    # --- Hardware Setup ---
    laser.setlidaropt(ydlidar.LidarPropSerialPort, "/dev/ttyUSB0")
    laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 128000)
    laser.setlidaropt(ydlidar.LidarPropLidarType, 18) # Using the 18 'fix' we found
    laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)

    # Initialize
    if not laser.initialize():
        print("Failed to initialize LiDAR")
        return
    if not laser.turnOn():
        print("Failed to start motor")
        return

    scan = ydlidar.LaserScan()
    print("Monitoring distance... Press Ctrl+C to stop.")

    try:
        while True:
            if laser.doProcessSimple(scan):
                distance = get_distance(scan)
                if distance is not None:
                    print(f"WALL DETECTED! Closest Distance: {distance:.2f} in")
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        laser.turnOff()
        laser.disconnecting()

if __name__ == "__main__":
    main()