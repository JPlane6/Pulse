import time
import ydlidar
from RPLCD.i2c import CharLCD
import lcd_hello
import motorControl_Pi as motors
import lidarHelpers
from lidarHelpers import get_stable_distances
from datetime import datetime

# --- Constants ---
LIDAR_PORT              = "/dev/lidar"
OBSTACLE_THRESHOLD_CM   = 50   # CHANGED: was 35, more braking distance at higher voltage
OPENING_INCREASE_CM     = 12   # CHANGED: was 20, catch openings earlier

# --- Turn-and-enter constants (measure these physically) ---
TURN_DURATION_SEC       = 1    # time in seconds for a 90° turn
ENTER_SPEED             = 155  # CHANGED: was 170, slowed down so LiDAR can catch openings
ENTER_DURATION_SEC      = 1    # how long to drive into the room
ENTER_ROOM_THRESHOLD_CM = 45   # minimum increase to consider it an opening


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


def update_lcd_fast(lcd, left_cm, right_cm, front_cm, status_line):
    lcd.cursor_pos = (1, 0)
    lcd.write_string(f"R:{right_cm:5.1f} L:{left_cm:5.1f}cm".ljust(20))
    lcd.cursor_pos = (2, 0)
    front_str = f"F:{front_cm:5.1f}cm" if front_cm is not None else "F: ---  cm"
    lcd.write_string(front_str.ljust(20))
    lcd.cursor_pos = (3, 0)
    lcd.write_string(status_line.ljust(20)[:20])


def log_distances(front_cm, left_cm, right_cm):
    ts = datetime.now().strftime("%H:%M:%S")
    front_str = f"{front_cm:.1f}" if front_cm is not None else "---"
    print(f"[{ts}] F:{front_str}cm L:{left_cm:.1f}cm R:{right_cm:.1f}cm")


def move_forward_until_obstacle(laser, lcd=None, threshold_cm=25, speed=ENTER_SPEED):
    motors.go()
    cmd = f"MOVE F {speed} 9999\n"
    motors.arduino.write(cmd.encode('utf-8'))
    print(f"[move_forward] Sent: {cmd.strip()}")
    motors.wait_for("MOVING", timeout=3)

    scan = ydlidar.LaserScan()
    print(f"[move_forward] Moving forward until obstacle at {threshold_cm}cm...")

    while True:
        front_cm, left_cm, right_cm = get_stable_distances(laser, scan)
        log_distances(front_cm, left_cm, right_cm)

        if lcd:
            update_lcd(lcd, left_cm, right_cm, front_cm, "MOVING FWD...")

        if front_cm is not None:
            print(f"[move_forward] Front: {front_cm:.1f}cm", end='\r')
            if front_cm <= threshold_cm:
                motors.stop()
                motors.go()
                print(f"\n[move_forward] STOPPED - Obstacle at {front_cm:.1f}cm")
                if lcd:
                    update_lcd(lcd, left_cm, right_cm, front_cm, "OBSTACLE DETECTED")
                return front_cm

        time.sleep(0.05)


def enter_room(side, laser, lcd=None):
    """
    Stop, turn 90 degrees toward the opening, then drive forward into the room.
    side: 'L' or 'R'
    laser: LiDAR object for distance measurement
    """
    print(f"[enter_room] Turning {side} into opening...")
    if lcd:
        lcd.cursor_pos = (3, 0)
        lcd.write_string(f"TURNING {side}...".ljust(20)[:20])

    motors.go()
    motors.turn(side, TURN_DURATION_SEC)

    print(f"[enter_room] Driving into room...")
    if lcd:
        lcd.cursor_pos = (3, 0)
        lcd.write_string("ENTERING ROOM...".ljust(20)[:20])

    motors.moveUntilThreshold('F', ENTER_ROOM_THRESHOLD_CM, laser, ENTER_SPEED)

    print(f"[enter_room] Entered room.")


def main():
    try:
        lcd = CharLCD("PCF8574", 0x27, cols=20, rows=4)
    except Exception:
        print("[main] ERROR: Could not connect to LCD.")
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

    prev_left_cm = None
    prev_right_cm = None
    prev_front_cm = None  # Track previous front distance for sudden-loss detection
    last_turn_time = 0
    TURN_COOLDOWN = 3.0
    motor_state = "IDLE"

    print("=" * 60)
    print("ROBOT NAVIGATION - READY")
    print(f"  - Front obstacle detection: 35-300cm (stops at {OBSTACLE_THRESHOLD_CM}cm)")
    print(f"  - Left/Right corridor walls: 20-300cm")
    print(f"  - Room opening threshold: +{OPENING_INCREASE_CM}cm increase")
    print(f"  - Turn cooldown: {TURN_COOLDOWN}s")
    print("=" * 60)

    try:
        while True:
            # Fast single-scan check for immediate obstacle detection
            if laser.doProcessSimple(scan):
                front_cm_fast, _, _ = lidarHelpers.get_all_distances(scan, debug=False)  # Debug disabled - working perfectly

                # CRITICAL: Check for obstacle IMMEDIATELY (no averaging delay)
                if front_cm_fast is not None and front_cm_fast < OBSTACLE_THRESHOLD_CM:
                    if motor_state != "STOPPED":
                        motors.stop()
                        motor_state = "STOPPED"
                        print(f"[main] EMERGENCY STOP - Obstacle at {front_cm_fast:.1f}cm")

                    # Get stable readings for display
                    front_cm, left_cm, right_cm = get_stable_distances(laser, scan, num_scans=3)
                    log_distances(front_cm, left_cm, right_cm)
                    update_lcd(lcd, left_cm, right_cm, front_cm, "OBSTACLE - STOPPED")

                    # Wait until obstacle clears
                    while True:
                        if laser.doProcessSimple(scan):
                            front_check, _, _ = lidarHelpers.get_all_distances(scan)

                            if front_check is None or front_check >= OBSTACLE_THRESHOLD_CM:
                                print("[main] Obstacle cleared — resuming.")
                                motors.go()
                                cmd = f"MOVE F {ENTER_SPEED} 9999\n"
                                motors.arduino.write(cmd.encode('utf-8'))
                                motors.wait_for("MOVING", timeout=3)
                                motor_state = "MOVING"
                                prev_front_cm = None  # Reset tracking
                                break

                            # Update display while waiting
                            if laser.doProcessSimple(scan):
                                front_cm, left_cm, right_cm = get_stable_distances(laser, scan, num_scans=2)
                                log_distances(front_cm, left_cm, right_cm)
                                update_lcd(lcd, left_cm, right_cm, front_cm, "OBSTACLE - STOPPED")
                    continue  # Restart main loop after clearing obstacle

                # FAILSAFE: If front reading suddenly disappears, obstacle likely very close - STOP!
                elif front_cm_fast is None and prev_front_cm is not None and prev_front_cm < OBSTACLE_THRESHOLD_CM:
                    if motor_state != "STOPPED":
                        motors.stop()
                        motor_state = "STOPPED"
                        print(f"[main] EMERGENCY STOP - Obstacle too close (lost detection at {prev_front_cm:.1f}cm)")

                    # Get stable readings for display
                    front_cm, left_cm, right_cm = get_stable_distances(laser, scan, num_scans=2)
                    log_distances(front_cm, left_cm, right_cm)
                    update_lcd(lcd, left_cm, right_cm, front_cm, "OBSTACLE - STOPPED")

                    # Wait until obstacle clears
                    while True:
                        if laser.doProcessSimple(scan):
                            front_check, _, _ = lidarHelpers.get_all_distances(scan)

                            if front_check is not None and front_check >= OBSTACLE_THRESHOLD_CM:
                                print("[main] Obstacle cleared — resuming.")
                                motors.go()
                                cmd = f"MOVE F {ENTER_SPEED} 9999\n"
                                motors.arduino.write(cmd.encode('utf-8'))
                                motors.wait_for("MOVING", timeout=3)
                                motor_state = "MOVING"
                                prev_front_cm = None
                                break

                            time.sleep(0.1)
                    continue

                # Update previous front for next iteration
                prev_front_cm = front_cm_fast

            # Get stable distances for normal operation (opening detection, display)
            front_cm, left_cm, right_cm = get_stable_distances(laser, scan, num_scans=2)
            log_distances(front_cm, left_cm, right_cm)

            # PRIORITY 2: LEFT OPENING
            if (prev_left_cm is not None and
                    left_cm < 999.0 and
                    left_cm - prev_left_cm > OPENING_INCREASE_CM and
                    time.time() - last_turn_time > TURN_COOLDOWN):

                print(f"[OPENING] LEFT room detected! Distance increased {left_cm - prev_left_cm:.0f}cm ({prev_left_cm:.0f}→{left_cm:.0f}cm)")
                print(f"[main] Opening on LEFT — stopping and entering | L:{left_cm:.1f} prev:{prev_left_cm:.1f}")
                motors.stop()
                update_lcd(lcd, left_cm, right_cm, front_cm, "OPENING LEFT!")
                time.sleep(0.3)

                enter_room('L', laser, lcd)

                prev_left_cm = left_cm
                prev_right_cm = right_cm
                last_turn_time = time.time()
                motor_state = "MOVING"

            # PRIORITY 3: RIGHT OPENING
            elif (prev_right_cm is not None and
                    right_cm < 999.0 and
                    right_cm - prev_right_cm > OPENING_INCREASE_CM and
                    time.time() - last_turn_time > TURN_COOLDOWN):

                print(f"[OPENING] RIGHT room detected! Distance increased {right_cm - prev_right_cm:.0f}cm ({prev_right_cm:.0f}→{right_cm:.0f}cm)")
                print(f"[main] Opening on RIGHT — stopping and entering | R:{right_cm:.1f} prev:{prev_right_cm:.1f}")
                motors.stop()
                update_lcd(lcd, left_cm, right_cm, front_cm, "OPENING RIGHT!")
                time.sleep(0.3)

                enter_room('R', laser, lcd)

                prev_right_cm = right_cm
                prev_left_cm = left_cm
                last_turn_time = time.time()
                motor_state = "MOVING"

            # PRIORITY 4: DEFAULT — go straight
            else:
                update_lcd(lcd, left_cm, right_cm, front_cm, "GOING STRAIGHT")
                if motor_state != "MOVING":
                    motors.go()
                    cmd = f"MOVE F {ENTER_SPEED} 9999\n"
                    motors.arduino.write(cmd.encode('utf-8'))
                    print("[main] Starting forward motion")
                    motors.wait_for("MOVING", timeout=3)
                    motor_state = "MOVING"

            if left_cm < 999.0:
                prev_left_cm = left_cm
            if right_cm < 999.0:
                prev_right_cm = right_cm

            # No sleep - check obstacles as fast as possible
            # time.sleep(0.05)

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