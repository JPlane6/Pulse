import ydlidar
import math

# ═══════════════════════════════════════════════════════════════════
#  DETECTION ZONE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
LIDAR_PORT   = "/dev/ttyUSB0"
FRONT_ANGLE  = 63    # Front detection center (degrees)
RIGHT_ANGLE  = 150   # Right detection center (degrees)
LEFT_ANGLE   = 340   # Left detection center (degrees)
FRONT_CONE   = 5     # Front detection range: 63±5° = 58-68°
SIDE_CONE    = 5     # Side detection range: L=335-345°, R=145-155°
MIN_DISTANCE_CM = 10.0   # Ignore anything closer than 10cm
MAX_DISTANCE_CM = 300.0  # Ignore anything farther than 300cm

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

def _get_angular_distance(angle_deg, center):
    """Calculate angular distance from center, handling 0/360 wrap."""
    diff = (angle_deg - center + 360) % 360
    if diff > 180:
        diff -= 360
    return abs(diff)


# ═══════════════════════════════════════════════════════════════════
#  SINGLE-PASS ZONE DETECTION (prevents cross-zone contamination)
# ═══════════════════════════════════════════════════════════════════
def get_all_distances(scan):
    """
    Process LiDAR scan in ONE PASS to prevent zone bleeding.
    Each point is assigned to the zone whose center it's closest to.

    Returns: (front_cm, left_cm, right_cm)
    - front_cm: Closest distance in front zone, or None
    - left_cm: Closest distance in left zone, or 999.0
    - right_cm: Closest distance in right zone, or 999.0
    """
    front_closest = None
    left_closest = None
    right_closest = None

    for point in scan.points:
        angle_deg = math.degrees(point.angle)
        if angle_deg < 0:
            angle_deg += 360

        dist_cm = m_to_cm(point.range)
        if dist_cm <= MIN_DISTANCE_CM or dist_cm > MAX_DISTANCE_CM:
            continue

        in_front = _angle_in_cone(angle_deg, FRONT_ANGLE, FRONT_CONE)
        in_left  = _angle_in_cone(angle_deg, LEFT_ANGLE,  SIDE_CONE)
        in_right = _angle_in_cone(angle_deg, RIGHT_ANGLE, SIDE_CONE)

        zone_matches = sum([in_front, in_left, in_right])

        if zone_matches == 0:
            continue
        elif zone_matches == 1:
            if in_front:
                if front_closest is None or dist_cm < front_closest:
                    front_closest = dist_cm
            elif in_left:
                if left_closest is None or dist_cm < left_closest:
                    left_closest = dist_cm
            elif in_right:
                if right_closest is None or dist_cm < right_closest:
                    right_closest = dist_cm
        else:
            distances = []
            if in_front:
                distances.append(('F', _get_angular_distance(angle_deg, FRONT_ANGLE)))
            if in_left:
                distances.append(('L', _get_angular_distance(angle_deg, LEFT_ANGLE)))
            if in_right:
                distances.append(('R', _get_angular_distance(angle_deg, RIGHT_ANGLE)))

            closest_zone = min(distances, key=lambda x: x[1])[0]

            if closest_zone == 'F':
                if front_closest is None or dist_cm < front_closest:
                    front_closest = dist_cm
            elif closest_zone == 'L':
                if left_closest is None or dist_cm < left_closest:
                    left_closest = dist_cm
            elif closest_zone == 'R':
                if right_closest is None or dist_cm < right_closest:
                    right_closest = dist_cm

    return front_closest, left_closest or 999.0, right_closest or 999.0


# ═══════════════════════════════════════════════════════════════════
#  AVERAGED SCAN (noise reduction)
# ═══════════════════════════════════════════════════════════════════
def get_stable_distances(laser, scan, num_scans=5):
    """Average N scans to reduce noise. Drop-in replacement for get_all_distances."""
    front_readings, left_readings, right_readings = [], [], []

    for _ in range(num_scans):
        if laser.doProcessSimple(scan):
            f, l, r = get_all_distances(scan)
            if f is not None:
                front_readings.append(f)
            if l < 999.0:
                left_readings.append(l)
            if r < 999.0:
                right_readings.append(r)

    front = sum(front_readings) / len(front_readings) if front_readings else None
    left  = sum(left_readings)  / len(left_readings)  if left_readings  else 999.0
    right = sum(right_readings) / len(right_readings) if right_readings else 999.0

    return front, left, right


# ═══════════════════════════════════════════════════════════════════
#  BACKWARD COMPATIBILITY WRAPPERS
# ═══════════════════════════════════════════════════════════════════
def get_front_distance(scan):
    """Get front distance - calls unified detector."""
    front_cm, _, _ = get_all_distances(scan)
    return front_cm

def get_left_right_distances_cm(scan):
    """Get left/right distances - calls unified detector."""
    _, left_cm, right_cm = get_all_distances(scan)
    return left_cm, right_cm

def is_obstacle_ahead(laser, scan, threshold_cm=30):
    """Uses averaged distances for accuracy."""
    front_dist, _, _ = get_stable_distances(laser, scan)
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