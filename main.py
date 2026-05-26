#!/usr/bin/env python3
"""
Hospital Gallery Navigation Robot
===================================
Navigates corridor, detects room openings, enters room,
then launches client.py to begin the patient assessment.
"""

import time
import subprocess   # ← ADDED: to launch client.py after entering room
import ydlidar
from RPLCD.i2c import CharLCD
import lcd_hello
import motorControl_Pi as motors
import lidarHelpers
from lidarHelpers import get_stable_distances
from datetime import datetime

# --- Constants --- (UNCHANGED)
LIDAR_PORT              = "/dev/lidar"
OBSTACLE_THRESHOLD_CM   = 45
OPENING_INCREASE_CM     = 12

# --- Turn-and-enter constants (UNCHANGED) ---
TURN_DURATION_SEC       = 1  # Adjust as needed for a 90° turn at your speed
ENTER_SPEED             = 100
ENTER_DURATION_SEC      = 1
ENTER_ROOM_THRESHOLD_CM = 45


# ═══════════════════════════════════════════════════════════════════
#  LCD HELPERS (UNCHANGED)
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
#  MOVEMENT HELPERS (UNCHANGED)
# ═══════════════════════════════════════════════════════════════════

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
    Stop, turn 90° toward the opening
    side: 'L' or 'R'
    """
    print(f"[enter_room] Turning {side} into opening...")
    if lcd:
        lcd.cursor_pos = (3, 0)
        lcd.write_string(f"TURNING {side}...".ljust(20)[:20])

    motors.go()
    motors.turn(side, TURN_DURATION_SEC)


    print(f"[enter_room] Entered room.")


# ═══════════════════════════════════════════════════════════════════
#  PATIENT ASSESSMENT TRIGGER
#  ← NEW: called after robot enters a room
# ═══════════════════════════════════════════════════════════════════

def start_patient_assessment(lcd=None):
    """
    Launch client.py as a subprocess to begin the patient triage session.
    This blocks until the full assessment is complete before navigation resumes.
    The LCD shows ASSESSING while it runs.
    """
    print("[assessment] Launching patient assessment (client.py)...")

    if lcd:
        lcd.clear()
        lcd.cursor_pos = (0, 0)
        lcd.write_string("PATIENT".ljust(20))
        lcd.cursor_pos = (1, 0)
        lcd.write_string("ASSESSMENT".ljust(20))
        lcd.cursor_pos = (2, 0)
        lcd.write_string("IN PROGRESS...".ljust(20))

    # Run client.py and wait for it to finish before continuing navigation
    result = subprocess.run(["python3", "client.py"], check=False)

    if result.returncode == 0:
        print("[assessment] Assessment completed successfully.")
    else:
        print(f"[assessment] client.py exited with code {result.returncode} — continuing navigation.")

    if lcd:
        lcd.clear()
        lcd.cursor_pos = (0, 0)
        lcd.write_string("ASSESSMENT DONE".ljust(20))
        lcd.cursor_pos = (1, 0)
        lcd.write_string("RESUMING NAV...".ljust(20))
        time.sleep(2)


# ═══════════════════════════════════════════════════════════════════
#  MAIN CONTROL LOOP
# ═══════════════════════════════════════════════════════════════════

def main():
    try:
        lcd = CharLCD("PCF8574", 0x27, cols=20, rows=4)
    except Exception:
        print("[main] ERROR: Could not connect to LCD.")
        return

    lcd.clear()
    lcd_hello.hello()

    # --- LiDAR setup (UNCHANGED) ---
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

    prev_left_cm  = None
    prev_right_cm = None
    prev_front_cm = None
    last_turn_time = 0
    TURN_COOLDOWN  = 3.0
    motor_state    = "IDLE"

    print("=" * 60)
    print("ROBOT NAVIGATION - READY")
    print(f"  - Front obstacle detection: 35-300cm (stops at {OBSTACLE_THRESHOLD_CM}cm)")
    print(f"  - Left/Right corridor walls: 20-300cm")
    print(f"  - Room opening threshold: +{OPENING_INCREASE_CM}cm increase")
    print(f"  - Turn cooldown: {TURN_COOLDOWN}s")
    print("=" * 60)

    try:
        while True:
            # ─────────────────────────────────────────────────────────────
            # FAST SCAN: Check front obstacle first, every loop iteration
            # ─────────────────────────────────────────────────────────────
            if laser.doProcessSimple(scan):
                front_cm_fast, _, _ = lidarHelpers.get_all_distances(scan, debug=False)

                # PRIORITY 1: EMERGENCY STOP — obstacle too close
                if front_cm_fast is not None and front_cm_fast < OBSTACLE_THRESHOLD_CM:
                    if motor_state != "STOPPED":
                        motors.stop()
                        motor_state = "STOPPED"
                        print(f"[main] EMERGENCY STOP - Obstacle at {front_cm_fast:.1f}cm")

                    front_cm, left_cm, right_cm = get_stable_distances(laser, scan, num_scans=3)
                    log_distances(front_cm, left_cm, right_cm)
                    update_lcd(lcd, left_cm, right_cm, front_cm, "OBSTACLE - STOPPED")

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
                                prev_front_cm = None
                                break
                            if laser.doProcessSimple(scan):
                                front_cm, left_cm, right_cm = get_stable_distances(laser, scan, num_scans=2)
                                log_distances(front_cm, left_cm, right_cm)
                                update_lcd(lcd, left_cm, right_cm, front_cm, "OBSTACLE - STOPPED")
                    continue

                # FAILSAFE: reading disappeared while close — likely too close to detect
                elif front_cm_fast is None and prev_front_cm is not None and prev_front_cm < OBSTACLE_THRESHOLD_CM:
                    if motor_state != "STOPPED":
                        motors.stop()
                        motor_state = "STOPPED"
                        print(f"[main] EMERGENCY STOP - Obstacle too close (lost at {prev_front_cm:.1f}cm)")

                    front_cm, left_cm, right_cm = get_stable_distances(laser, scan, num_scans=2)
                    log_distances(front_cm, left_cm, right_cm)
                    update_lcd(lcd, left_cm, right_cm, front_cm, "OBSTACLE - STOPPED")

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

                prev_front_cm = front_cm_fast

            # ─────────────────────────────────────────────────────────────
            # STABLE SCAN: used for opening detection and display
            # ─────────────────────────────────────────────────────────────
            front_cm, left_cm, right_cm = get_stable_distances(laser, scan, num_scans=2)
            log_distances(front_cm, left_cm, right_cm)

            # PRIORITY 2: LEFT OPENING DETECTED
            if (prev_left_cm is not None and
                    left_cm < 999.0 and
                    left_cm - prev_left_cm > OPENING_INCREASE_CM and
                    time.time() - last_turn_time > TURN_COOLDOWN):

                print(f"[OPENING] LEFT room detected! {left_cm - prev_left_cm:.0f}cm increase ({prev_left_cm:.0f}→{left_cm:.0f}cm)")
                motors.stop()
                update_lcd(lcd, left_cm, right_cm, front_cm, "OPENING LEFT!")
                time.sleep(0.3)

                enter_room('L', laser, lcd)

                # ← NEW: assessment starts here, blocks until done
                start_patient_assessment(lcd)

                prev_left_cm   = left_cm
                prev_right_cm  = right_cm
                last_turn_time = time.time()
                motor_state    = "MOVING"

            # PRIORITY 3: RIGHT OPENING DETECTED
            elif (prev_right_cm is not None and
                    right_cm < 999.0 and
                    right_cm - prev_right_cm > OPENING_INCREASE_CM and
                    time.time() - last_turn_time > TURN_COOLDOWN):

                print(f"[OPENING] RIGHT room detected! {right_cm - prev_right_cm:.0f}cm increase ({prev_right_cm:.0f}→{right_cm:.0f}cm)")
                motors.stop()
                update_lcd(lcd, left_cm, right_cm, front_cm, "OPENING RIGHT!")
                time.sleep(0.3)

                enter_room('R', laser, lcd)

                # ← NEW: assessment starts here, blocks until done
                start_patient_assessment(lcd)

                prev_right_cm  = right_cm
                prev_left_cm   = left_cm
                last_turn_time = time.time()
                motor_state    = "MOVING"

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