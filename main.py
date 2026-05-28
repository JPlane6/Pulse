#!/usr/bin/env python3
"""
Hospital Gallery Navigation Robot
===================================
Navigates corridor, detects room openings, enters room,
then launches client.py to begin the patient assessment.

BOOT ORDER (parallel):
  Thread A — LCD hello animation
  Thread B — Arduino connect (replaces blocking sleep with READY handshake)
  Thread C — LiDAR initialize + turnOn
All three run simultaneously. Main blocks until all three finish.
"""

import json
import os
import time
import threading
import subprocess
import ydlidar
from RPLCD.i2c import CharLCD
import lcd_hello
import motorControl_Pi as motors
import lidarHelpers
from lidarHelpers import get_stable_distances
from datetime import datetime

PATIENT_INFO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules", "logs", "patientINFO.json")

LIDAR_PORT              = "/dev/lidar"
OBSTACLE_THRESHOLD_CM   = 45
OPENING_INCREASE_CM     = 12
TURN_DURATION_SEC       = 0.8
ENTER_SPEED             = 85
ENTER_DURATION_SEC      = 1
ENTER_ROOM_THRESHOLD_CM = 45


# ═══════════════════════════════════════════════════════════════════
#  LCD HELPERS
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
#  MOVEMENT HELPERS
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
    print(f"[enter_room] Turning {side} into opening...")
    if lcd:
        lcd.cursor_pos = (3, 0)
        lcd.write_string(f"TURNING {side}...".ljust(20)[:20])

    motors.go()
    motors.turn(side, TURN_DURATION_SEC)
    print(f"[enter_room] Entered room.")


# ═══════════════════════════════════════════════════════════════════
#  PATIENT ASSESSMENT TRIGGER
# ═══════════════════════════════════════════════════════════════════

def start_patient_assessment(lcd=None):
    print("[assessment] Launching patient assessment (client.py)...")

    if lcd:
        lcd.clear()
        lcd.cursor_pos = (0, 0); lcd.write_string("PATIENT".ljust(20))
        lcd.cursor_pos = (1, 0); lcd.write_string("ASSESSMENT".ljust(20))
        lcd.cursor_pos = (2, 0); lcd.write_string("IN PROGRESS...".ljust(20))

    result = subprocess.run(["python3", "client.py"], check=False)

    if result.returncode == 0:
        print("[assessment] Assessment completed successfully.")
    else:
        print(f"[assessment] client.py exited with code {result.returncode}.")

    status  = "COMPLETE"
    flagged = False

    try:
        if os.path.exists(PATIENT_INFO_PATH):
            with open(PATIENT_INFO_PATH, "r") as f:
                all_records = json.load(f)
            if all_records:
                last    = all_records[-1]
                status  = last.get("triage", {}).get("final_status", "COMPLETE")
                flagged = last.get("triage", {}).get("flagged_urgent", False)
    except Exception as e:
        print(f"[assessment] Could not read triage result: {e}")

    motors.stop()
    print("[assessment] Navigation halted. Displaying triage result indefinitely.")

    if lcd:
        while True:
            lcd.clear()
            lcd.cursor_pos = (0, 0); lcd.write_string("== TRIAGE STATUS ==")
            lcd.cursor_pos = (1, 0); lcd.write_string(f"Result: {status}".ljust(20)[:20])
            lcd.cursor_pos = (2, 0)
            if flagged:
                lcd.write_string("!! NURSE ALERTED !!".ljust(20))
            else:
                lcd.write_string("Assessment Done".ljust(20))
            lcd.cursor_pos = (3, 0); lcd.write_string("Session Complete".ljust(20))
            time.sleep(5)
    else:
        while True:
            print(f"[assessment] Triage result: {status} | Urgent: {flagged}")
            time.sleep(10)


# ═══════════════════════════════════════════════════════════════════
#  PARALLEL BOOT
# ═══════════════════════════════════════════════════════════════════

def _boot_lcd(results, errors):
    """Thread A: LCD init + hello animation."""
    try:
        lcd = CharLCD("PCF8574", 0x27, cols=20, rows=4)
        lcd.clear()
        lcd_hello.hello()          # runs the 5-second animation
        results["lcd"] = lcd
        print("[boot] LCD ready")
    except Exception as e:
        errors["lcd"] = e
        print(f"[boot] LCD failed: {e}")


def _boot_arduino(results, errors):
    """Thread B: Arduino serial connect (READY handshake, no fixed sleep)."""
    try:
        motors.connect()           # our new lazy-init connect()
        results["arduino"] = True
        print("[boot] Arduino ready")
    except Exception as e:
        errors["arduino"] = e
        print(f"[boot] Arduino failed: {e}")


def _boot_lidar(results, errors):
    """Thread C: LiDAR initialize + motor spin-up."""
    try:
        laser = ydlidar.CYdLidar()
        laser.setlidaropt(ydlidar.LidarPropSerialPort, LIDAR_PORT)
        laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 128000)
        laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
        laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
        laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)
        laser.setlidaropt(ydlidar.LidarPropSampleRate, 5)
        laser.setlidaropt(ydlidar.LidarPropScanFrequency, 12.0)

        if not laser.initialize():
            raise RuntimeError("LiDAR initialize() failed")
        if not laser.turnOn():
            raise RuntimeError("LiDAR turnOn() failed")

        results["laser"] = laser
        print("[boot] LiDAR ready")
    except Exception as e:
        errors["lidar"] = e
        print(f"[boot] LiDAR failed: {e}")


def parallel_boot():
    """
    Kick off LCD, Arduino, and LiDAR init simultaneously.
    Returns (lcd, laser) once all three threads finish.
    Raises RuntimeError if any critical component failed.
    """
    results = {}
    errors  = {}

    t_lcd     = threading.Thread(target=_boot_lcd,     args=(results, errors), daemon=True)
    t_arduino = threading.Thread(target=_boot_arduino, args=(results, errors), daemon=True)
    t_lidar   = threading.Thread(target=_boot_lidar,   args=(results, errors), daemon=True)

    boot_start = time.time()
    t_lcd.start()
    t_arduino.start()
    t_lidar.start()

    # Wait for all three — LCD hello is 5s so that's the natural gate
    t_lcd.join()
    t_arduino.join()
    t_lidar.join()

    elapsed = time.time() - boot_start
    print(f"[boot] All components ready in {elapsed:.1f}s")

    if "lcd" in errors:
        raise RuntimeError(f"LCD failed: {errors['lcd']}")
    if "arduino" in errors:
        raise RuntimeError(f"Arduino failed: {errors['arduino']}")
    if "lidar" in errors:
        lcd = results.get("lcd")
        if lcd:
            lcd.clear()
            lcd.write_string("LIDAR INIT ERROR".ljust(20))
            lcd.cursor_pos = (1, 0)
            lcd.write_string(LIDAR_PORT.ljust(20)[:20])
        raise RuntimeError(f"LiDAR failed: {errors['lidar']}")

    return results["lcd"], results["laser"]


# ═══════════════════════════════════════════════════════════════════
#  MAIN CONTROL LOOP
# ═══════════════════════════════════════════════════════════════════

def main():
    # ── Parallel boot ──────────────────────────────────────────────
    try:
        lcd, laser = parallel_boot()
    except RuntimeError as e:
        print(f"[main] BOOT FAILED: {e}")
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
            # ── FAST SCAN: front obstacle check every loop ─────────
            if laser.doProcessSimple(scan):
                front_cm_fast, _, _ = lidarHelpers.get_all_distances(scan, debug=False)

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
                                motor_state   = "MOVING"
                                prev_front_cm = None
                                break
                            if laser.doProcessSimple(scan):
                                front_cm, left_cm, right_cm = get_stable_distances(laser, scan, num_scans=2)
                                log_distances(front_cm, left_cm, right_cm)
                                update_lcd(lcd, left_cm, right_cm, front_cm, "OBSTACLE - STOPPED")
                    continue

                elif front_cm_fast is None and prev_front_cm is not None and prev_front_cm < OBSTACLE_THRESHOLD_CM:
                    if motor_state != "STOPPED":
                        motors.stop()
                        motor_state   = "STOPPED"
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
                                motor_state   = "MOVING"
                                prev_front_cm = None
                                break
                            time.sleep(0.1)
                    continue

                prev_front_cm = front_cm_fast

            # ── STABLE SCAN: opening detection + display ───────────
            front_cm, left_cm, right_cm = get_stable_distances(laser, scan, num_scans=2)
            log_distances(front_cm, left_cm, right_cm)

            if (prev_left_cm is not None and
                    left_cm < 999.0 and
                    left_cm - prev_left_cm > OPENING_INCREASE_CM and
                    time.time() - last_turn_time > TURN_COOLDOWN):

                print(f"[OPENING] LEFT room detected! {left_cm - prev_left_cm:.0f}cm increase ({prev_left_cm:.0f}→{left_cm:.0f}cm)")
                motors.stop()
                update_lcd(lcd, left_cm, right_cm, front_cm, "OPENING LEFT!")
                time.sleep(0.3)
                enter_room('L', laser, lcd)
                start_patient_assessment(lcd)

                prev_left_cm   = left_cm
                prev_right_cm  = right_cm
                last_turn_time = time.time()
                motor_state    = "MOVING"

            elif (prev_right_cm is not None and
                    right_cm < 999.0 and
                    right_cm - prev_right_cm > OPENING_INCREASE_CM and
                    time.time() - last_turn_time > TURN_COOLDOWN):

                print(f"[OPENING] RIGHT room detected! {right_cm - prev_right_cm:.0f}cm increase ({prev_right_cm:.0f}→{right_cm:.0f}cm)")
                motors.stop()
                update_lcd(lcd, left_cm, right_cm, front_cm, "OPENING RIGHT!")
                time.sleep(0.3)
                enter_room('R', laser, lcd)
                start_patient_assessment(lcd)

                prev_right_cm  = right_cm
                prev_left_cm   = left_cm
                last_turn_time = time.time()
                motor_state    = "MOVING"

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