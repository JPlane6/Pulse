import time
import math
import ydlidar
from RPLCD.i2c import CharLCD
import lcd_hello
import motorControl_Pi as motors
import lidarHelpers as lidarHelpers

LIDAR_PORT = "/dev/ttyUSB0"

def main():
    try:
        lcd = CharLCD("PCF8574", 0x27, cols=20, rows=4)
    except Exception:
        return

    lcd.clear()
    lcd_hello.hello()

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

            if (lidarHelpers.is_obstcle_ahead(scan, 30)):
                motors.stop()
            else:
                motors.go()
                motors.moveUntilThreshold("FORWARD", 200, 30, laser)

            if laser.doProcessSimple(scan):
                left_cm, right_cm = lidarHelpers.get_left_right_distances_cm(scan)
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
        motors.stop()
    finally:
        laser.turnOff()
        laser.disconnecting()
        lcd.clear()
        lcd.write_string("SYSTEM IDLE".ljust(20))
        motors.go()


if __name__ == "__main__":
    main()