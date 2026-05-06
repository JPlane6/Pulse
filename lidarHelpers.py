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

def is_obstacle_ahead(scan, threshold_cm=30):
    front_dist = get_front_distance(scan)
    return front_dist is not None and front_dist < threshold_cm

def get_left_right_distances_cm(scan, side_window_deg=20):
    left_closest = None
    right_closest = None

    LEFT_CENTER = 150
    RIGHT_CENTER = 350

    for point in scan.points:
        angle_deg = math.degrees(point.angle)
        if angle_deg < 0:
            angle_deg += 360

        dist_m = point.range
        if dist_m <= 0.10:
            continue

        dist_cm = dist_m * 100.0

        if (LEFT_CENTER - side_window_deg) <= angle_deg <= (LEFT_CENTER + side_window_deg):
            if left_closest is None or dist_cm < left_closest:
                left_closest = dist_cm

        if (RIGHT_CENTER - side_window_deg) <= angle_deg <= (RIGHT_CENTER + side_window_deg):
            if right_closest is None or dist_cm < right_closest:
                right_closest = dist_cm

    return left_closest or 999.0, right_closest or 999.0

def get_turn_direction(scan):
    left_cm, right_cm = get_left_right_distances_cm(scan)
    if left_cm > right_cm:
        return 'L'
    elif right_cm > left_cm:
        return 'R'
    else:
        return 'R'  # default fallback

def get_all_distance(scan):
    closest = None
    for point in scan.points:
        dist_cm = m_to_cm(point.range)
        if 3.0 < dist_cm <= 300.0:
            if closest is None or dist_cm < closest:
                closest = dist_cm
    return closest