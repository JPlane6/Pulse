import time
import math
import ydlidar
from RPLCD.i2c import CharLCD
import lcd_hello

LIDAR_PORT = "/dev/ttyUSB0"


def get_left_right_distances_cm(scan, side_window_deg=20):
    left_closest = None
    right_closest = None

    for point in scan.points:
        angle_deg = math.degrees(point.angle)
        if angle_deg < 0:
            angle_deg += 360

        dist_m = point.range
        if dist_m <= 0.10:
            continue

        dist_cm = dist_m * 100.0
        adjusted_angle = 150
        offsetR = 20

        if (adjusted_angle - side_window_deg) <= angle_deg <= (adjusted_angle + side_window_deg):
            if left_closest is None or dist_cm < left_closest:
                left_closest = dist_cm

        if (350 - side_window_deg) <= angle_deg <= (350 + side_window_deg):
            if right_closest is None or dist_cm < right_closest:
                right_closest = dist_cm
    return left_closest or 999.0, right_closest or 999.0


def main():
    try:
        lcd = CharLCD("PCF8574", 0x27, cols=20, rows=4)
    except Exception:
        return

    lcd.clear()
    # lcd_hello.hello()

    laser = ydlidar.CYdLidar()
    laser.setlidaropt(ydlidar.LidarPropSerialPort, LIDAR_PORT)
    laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 128000)
    laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
    laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
    laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)
    laser.setlidaropt(ydlidar.LidarPropSampleRate, 5)
    laser.setlidaropt(ydlidar.LidarPropScanFrequency, 12.0)


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