import time
import ydlidar
from RPLCD.i2c import CharLCD
import lcd_hello
import motorControl_Pi as motors
import lidarHelpers
# from modules import vision, tts, patient, stt

# --- Constants ---
LIDAR_PORT              = "/dev/ttyUSB0"
OBSTACLE_THRESHOLD_CM   = 20
OPENING_INCREASE_CM     = 50  # Increase in side distance to detect opening/doorway


# ------------------------------------------------------------------ #
#  LCD helper — Row 1: header, Row 2: R/L, Row 3: Front, Row 4: status
# ------------------------------------------------------------------ #
def update_lcd(lcd, left_cm, right_cm, front_cm, status_line):
    lcd.cursor_pos = (0, 0)
    lcd.write_string("R/L/F DISTANCES".ljust(20))
    lcd.cursor_pos = (1, 0)
    lcd.write_string(f"R:{right_cm:5.1f} L:{left_cm:5.1f}cm".ljust(20))
    lcd.cursor_pos = (2, 0)
    front_str = f"F:{front_cm:5.1f}cm" if front_cm is not None else "F: ---  cm"
    lcd.write_string(front_str.ljust(20))
    lcd.cursor_pos = (3, 0)
    lcd.write_string(status_line.ljust(20)[:20])


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #
def main():
    # --- LCD init ---
    try:
        lcd = CharLCD("PCF8574", 0x27, cols=20, rows=4)
    except Exception:
        print("[main] ERROR: Could not connect to LCD.")
        return

    lcd.clear()
    lcd_hello.hello()

    # --- LiDAR init ---
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

    # Track previous distances to detect openings
    prev_left_cm = None
    prev_right_cm = None

    # ---------------------------------------------------------------- #
    #  Main loop — simplified gallery navigation                       #
    # ---------------------------------------------------------------- #
    try:
        while True:
            if not laser.doProcessSimple(scan):
                time.sleep(0.05)
                continue

            left_cm, right_cm = lidarHelpers.get_left_right_distances_cm(scan)
            front_cm = lidarHelpers.get_front_distance(scan)

            # -------------------------------------------------------- #
            #  Priority 1 — obstacle in front: stop and wait          #
            # -------------------------------------------------------- #
            if lidarHelpers.is_obstacle_ahead(scan, threshold_cm=OBSTACLE_THRESHOLD_CM):
                motors.stop()
                update_lcd(lcd, left_cm, right_cm, front_cm, "OBSTACLE - STOPPED")
                print(f"[main] OBSTACLE ahead — stopped | F:{front_cm} L:{left_cm:.1f} R:{right_cm:.1f}")

                # Wait until obstacle clears
                while True:
                    if laser.doProcessSimple(scan):
                        left_cm, right_cm = lidarHelpers.get_left_right_distances_cm(scan)
                        front_cm = lidarHelpers.get_front_distance(scan)
                        update_lcd(lcd, left_cm, right_cm, front_cm, "OBSTACLE - STOPPED")
                        
                        if not lidarHelpers.is_obstacle_ahead(scan, threshold_cm=OBSTACLE_THRESHOLD_CM):
                            print("[main] Obstacle cleared — resuming.")
                            update_lcd(lcd, left_cm, right_cm, front_cm, "RESUMING...")
                            time.sleep(0.5)
                            break
                    time.sleep(0.05)

            # -------------------------------------------------------- #
            #  Priority 2 — opening detected on left: turn left       #
            # -------------------------------------------------------- #
            elif (prev_left_cm is not None and 
                  left_cm < 999.0 and 
                  left_cm - prev_left_cm > OPENING_INCREASE_CM):
                
                motors.stop()
                print(f"[main] Opening on LEFT detected — turning | F:{front_cm} L:{left_cm:.1f} R:{right_cm:.1f}")
                
                # Turn left while continuously updating LCD
                turn_start = time.time()
                while time.time() - turn_start < 2:
                    motors.turn('L', 0.1)
                    if laser.doProcessSimple(scan):
                        left_cm, right_cm = lidarHelpers.get_left_right_distances_cm(scan)
                        front_cm = lidarHelpers.get_front_distance(scan)
                        update_lcd(lcd, left_cm, right_cm, front_cm, "TURNING LEFT")
                    time.sleep(0.05)
                
                motors.stop()
                time.sleep(0.3)

            # -------------------------------------------------------- #
            #  Priority 3 — opening detected on right: turn right     #
            # -------------------------------------------------------- #
            elif (prev_right_cm is not None and 
                  right_cm < 999.0 and 
                  right_cm - prev_right_cm > OPENING_INCREASE_CM):
                
                motors.stop()
                print(f"[main] Opening on RIGHT detected — turning | F:{front_cm} L:{left_cm:.1f} R:{right_cm:.1f}")
                
                # Turn right while continuously updating LCD
                turn_start = time.time()
                while time.time() - turn_start < 2:
                    motors.turn('R', 0.1)
                    if laser.doProcessSimple(scan):
                        left_cm, right_cm = lidarHelpers.get_left_right_distances_cm(scan)
                        front_cm = lidarHelpers.get_front_distance(scan)
                        update_lcd(lcd, left_cm, right_cm, front_cm, "TURNING RIGHT")
                    time.sleep(0.05)
                
                motors.stop()
                time.sleep(0.3)

            # -------------------------------------------------------- #
            #  Priority 4 — go straight                                #
            # -------------------------------------------------------- #
            else:
                update_lcd(lcd, left_cm, right_cm, front_cm, "GOING STRAIGHT")
                motors.go()

            # Update previous distances for next iteration
            if left_cm < 999.0:
                prev_left_cm = left_cm
            if right_cm < 999.0:
                prev_right_cm = right_cm

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("[main] Interrupted.")
        motors.stop()
    finally:
        laser.turnOff()
        laser.disconnecting()
        lcd.clear()
        lcd.write_string("SYSTEM IDLE".ljust(20))
        motors.go()


if __name__ == "__main__":
    main()