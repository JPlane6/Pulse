import ydlidar
import math

LIDAR_PORT = "/dev/ttyUSB0"

def m_to_cm(meters):
    return meters * 100.0

def init_lidar():
    laser = ydlidar.CYdLidar()
    laser.setlidaropt(ydlidar.LidarPropSerialPort, LIDAR_PORT)
    laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 128000)
    laser.setlidaropt(ydlidar.LidarPropLidarType, 18)
    laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)
    if not laser.initialize():
        print("Failed to initialize LiDAR")
        exit()
    if not laser.turnOn():
        print("Failed to start LiDAR motor")
        exit()
    return laser

def get_front_distance(scan, cone_deg=30):
    """Returns closest valid distance in cm within a forward cone, or None."""
    closest = None
    for point in scan.points:
        angle_deg = math.degrees(point.angle)
        in_front = angle_deg <= cone_deg or angle_deg >= (360 - cone_deg)
        if in_front:
            dist_cm = m_to_cm(point.range)
            if 3.0 < dist_cm <= 300.0:
                if closest is None or dist_cm < closest:
                    closest = dist_cm
    return closest

def get_all_distance(scan):
    """Returns closest valid distance in cm from full 360 scan, or None."""
    closest = None
    for point in scan.points:
        dist_cm = m_to_cm(point.range)
        if 3.0 < dist_cm <= 300.0:
            if closest is None or dist_cm < closest:
                closest = dist_cm
    return closest