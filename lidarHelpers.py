import ydlidar
import math

LIDAR_PORT   = "/dev/ttyUSB0"
FRONT_ANGLE  = 63    # confirmed from front wall scan
RIGHT_ANGLE  = 332   # swapped — was LEFT
LEFT_ANGLE   = 145   # swapped — was RIGHT
FRONT_CONE   = 20
SIDE_CONE    = 20

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

def _angle_in_cone(angle_deg, center, cone):
    """Check if angle is within ±cone degrees of center, handling 0/360 wrap."""
    diff = (angle_deg - center + 360) % 360
    if diff > 180:
        diff -= 360
    return abs(diff) <= cone

def get_front_distance(scan):
    closest = None
    for point in scan.points:
        angle_deg = math.degrees(point.angle)
        if angle_deg < 0:
            angle_deg += 360
        if _angle_in_cone(angle_deg, FRONT_ANGLE, FRONT_CONE):
            dist_cm = m_to_cm(point.range)
            if 3.0 < dist_cm <= 300.0:
                if closest is None or dist_cm < closest:
                    closest = dist_cm
    return closest

def get_left_right_distances_cm(scan):
    left_closest  = None
    right_closest = None
    for point in scan.points:
        angle_deg = math.degrees(point.angle)
        if angle_deg < 0:
            angle_deg += 360
        dist_m = point.range
        if dist_m <= 0.10:
            continue
        dist_cm = dist_m * 100.0
        if _angle_in_cone(angle_deg, LEFT_ANGLE, SIDE_CONE):
            if left_closest is None or dist_cm < left_closest:
                left_closest = dist_cm
        if _angle_in_cone(angle_deg, RIGHT_ANGLE, SIDE_CONE):
            if right_closest is None or dist_cm < right_closest:
                right_closest = dist_cm
    return left_closest or 999.0, right_closest or 999.0

def is_obstacle_ahead(scan, threshold_cm=30):
    front_dist = get_front_distance(scan)
    return front_dist is not None and front_dist < threshold_cm

def get_obstacle_direction(scan, threshold_cm=30):
    """Returns 'F', 'L', 'R', or None — checks each zone independently."""
    front_cm = get_front_distance(scan)
    left_cm, right_cm = get_left_right_distances_cm(scan)

    if front_cm is not None and front_cm < threshold_cm:
        return 'F'
    if left_cm < threshold_cm:
        return 'L'
    if right_cm < threshold_cm:
        return 'R'
    return None

def get_turn_direction(scan):
    left_cm, right_cm = get_left_right_distances_cm(scan)
    if left_cm > right_cm:
        return 'L'
    elif right_cm > left_cm:
        return 'R'
    else:
        return 'R'

def get_all_distance(scan):
    closest = None
    for point in scan.points:
        dist_cm = m_to_cm(point.range)
        if 3.0 < dist_cm <= 300.0:
            if closest is None or dist_cm < closest:
                closest = dist_cm
    return closest