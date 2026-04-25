import time
import math
import ydlidar
from RPLCD.i2c import CharLCD
import lcd_hello

LIDAR_PORT = "/dev/ttyUSB0"


def get_left_right_distances_cm(scan, side_window_deg=30):
    left_min = 90 - side_window_deg
    left_max = 90 + side_window_deg
    right_min = 270 - side_window_deg
    right_max = 270 + side_window_deg

    left_closest = None
    right_closest = None

    for point in scan.points:
        angle_deg = math.degrees(point.angle)
        dist_m = point.range
        if dist_m <= 0.10:
            continue

        dist_cm = dist_m * 100.0

        if left_min <= angle_deg <= left_max:
            if left_closest is None or dist_cm < left_closest:
                left_closest = dist_cm

        if right_min <= angle_deg <= right_max:
            if right_closest is None or dist_cm < right_closest:
                right_closest = dist_cm

    if left_closest is None:
        left_closest = 999.0
    if right_closest is None:
        right_closest = 999.0

    return left_closest, right_closest


def main():
    try:
        lcd = CharLCD("PCF8574", 0x27, cols=20, rows=4)
    except Exception:
        return

    lcd.clear()
    lcd_hello.hello()
    # lcd.write_string("PULSE SYSTEM".ljust(20))
    # lcd.cursor_pos = (1, 0)
    # lcd.write_string("INITIALIZING...".ljust(20))

    laser = ydlidar.CYdLidar()
    laser.setlidaropt(ydlidar.LidarPropSerialPort, LIDAR_PORT)
    laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 128000)
    laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
    laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
    laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)
    laser.setlidaropt(ydlidar.LidarPropSampleRate, 3)
    laser.setlidaropt(ydlidar.LidarPropScanFrequency, 5.0)

    time.sleep(2)

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

    try:
        while True:
            if laser.doProcessSimple(scan):
                left_cm, right_cm = get_left_right_distances_cm(scan)
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
        pass
    finally:
        laser.turnOff()
        laser.disconnecting()
        lcd.clear()
        lcd.write_string("SYSTEM IDLE".ljust(20))


if __name__ == "__main__":
    main()
