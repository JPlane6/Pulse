import time
import ydlidar
from RPLCD.i2c import CharLCD
import lcd_hello
import motorControl_Pi as motors
import lidarHelpers
# from modules import vision, tts, patient, stt

# --- Constants ---
LIDAR_PORT              = "/dev/ttyUSB0"
OBSTACLE_THRESHOLD_CM   = 30
WALL_NUDGE_THRESHOLD_CM = 20
WALL_NUDGE_DURATION_SEC = 0.3
DOOR_GAP_THRESHOLD_CM   = 80


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

    # ---------------------------------------------------------------- #
    #  Main loop                                                        #
    # ---------------------------------------------------------------- #
    try:
        while True:
            if not laser.doProcessSimple(scan):
                time.sleep(0.05)
                continue

            left_cm, right_cm = lidarHelpers.get_left_right_distances_cm(scan)
            front_cm = lidarHelpers.get_front_distance(scan)
            obs_dir  = lidarHelpers.get_obstacle_direction(scan, threshold_cm=OBSTACLE_THRESHOLD_CM)

            # -------------------------------------------------------- #
            #  Priority 1 — obstacle ahead: stop and wait             #
            # -------------------------------------------------------- #
            if lidarHelpers.is_obstacle_ahead(scan, threshold_cm=OBSTACLE_THRESHOLD_CM):
                motors.stop()
                motors.go()
                print(f"[main] OBSTACLE F — stopped | F:{front_cm} L:{left_cm:.1f} R:{right_cm:.1f}")

                while True:
                    obs_dir = lidarHelpers.get_obstacle_direction(scan, threshold_cm=OBSTACLE_THRESHOLD_CM)
                    dir_label = obs_dir if obs_dir else "F"
                    update_lcd(lcd, left_cm, right_cm, front_cm, f"OBSTACLE {dir_label}")
                    if laser.doProcessSimple(scan):
                        left_cm, right_cm = lidarHelpers.get_left_right_distances_cm(scan)
                        front_cm = lidarHelpers.get_front_distance(scan)
                        if not lidarHelpers.is_obstacle_ahead(scan, threshold_cm=OBSTACLE_THRESHOLD_CM):
                            print("[main] Obstacle cleared — resuming.")
                            update_lcd(lcd, left_cm, right_cm, front_cm, "RESUMING...")
                            time.sleep(1)
                            break
                    time.sleep(0.05)

            # -------------------------------------------------------- #
            #  Priority 2 — doorway detected: large gap on one side   #
            # -------------------------------------------------------- #
            elif (abs(left_cm - right_cm) > DOOR_GAP_THRESHOLD_CM
                  and left_cm < 999.0 and right_cm < 999.0):
                turn_dir  = lidarHelpers.get_turn_direction(scan)
                turn_word = "RIGHT" if turn_dir == "R" else "LEFT"

                update_lcd(lcd, left_cm, right_cm, front_cm, f"DOOR {turn_word}")
                print(f"[main] Doorway — turning {turn_word} | F:{front_cm} L:{left_cm:.1f} R:{right_cm:.1f}")

                motors.stop()
                motors.go()

                while True:
                    motors.turn(turn_dir, 1)
                    if laser.doProcessSimple(scan):
                        left_cm, right_cm = lidarHelpers.get_left_right_distances_cm(scan)
                        front_cm = lidarHelpers.get_front_distance(scan)
                        update_lcd(lcd, left_cm, right_cm, front_cm, f"DOOR {turn_word}")
                        if not lidarHelpers.is_obstacle_ahead(scan, threshold_cm=OBSTACLE_THRESHOLD_CM):
                            print("[main] Aligned into doorway — resuming.")
                            break

                motors.stop()
                motors.go()

            # -------------------------------------------------------- #
            #  Priority 3 — wall drift (small correction)             #
            # -------------------------------------------------------- #
            elif (abs(left_cm - right_cm) > WALL_NUDGE_THRESHOLD_CM
                  and left_cm < 999.0 and right_cm < 999.0):
                nudge_dir  = lidarHelpers.get_turn_direction(scan)
                nudge_word = "RIGHT" if nudge_dir == "R" else "LEFT"

                update_lcd(lcd, left_cm, right_cm, front_cm, f"NUDGE {nudge_word}")
                print(f"[main] Wall drift — nudging {nudge_word} | F:{front_cm} L:{left_cm:.1f} R:{right_cm:.1f}")
                motors.turn(nudge_dir, WALL_NUDGE_DURATION_SEC)

            # -------------------------------------------------------- #
            #  Priority 4 — all clear                                  #
            # -------------------------------------------------------- #
            else:
                update_lcd(lcd, left_cm, right_cm, front_cm, "RUNNING")
                motors.go()
                motors.moveUntilThreshold("F", OBSTACLE_THRESHOLD_CM, laser, 200)

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