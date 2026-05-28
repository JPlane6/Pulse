#!/usr/bin/env python3
"""
PULSE — Hospital Gallery Navigation Robot
Parallel boot, optimised LiDAR polling, no dead sleeps.
"""

import json
import os
import time
import threading
import subprocess
import ydlidar
from RPLCD.i2c import CharLCD
import motorControl_Pi as motors
import lidarHelpers
from lidarHelpers import get_stable_distances
from datetime import datetime

PATIENT_INFO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules", "logs", "patientINFO.json")

LIDAR_PORT            = "/dev/lidar"
OBSTACLE_THRESHOLD_CM = 45
OPENING_INCREASE_CM   = 12
TURN_DURATION_SEC     = 0.8
ENTER_SPEED           = 85
ENTER_ROOM_THRESHOLD  = 45


# ═══════════════════════════════════════════════════════════════════
#  LCD HELPERS
# ═══════════════════════════════════════════════════════════════════
def update_lcd(lcd, left_cm, right_cm, front_cm, status_line):
    if lcd is None: return
    lcd.cursor_pos = (0, 0); lcd.write_string("R/L/F DISTANCES".ljust(20))
    lcd.cursor_pos = (1, 0); lcd.write_string(f"R:{right_cm:5.1f} L:{left_cm:5.1f}cm".ljust(20))
    lcd.cursor_pos = (2, 0)
    front_str = f"F:{front_cm:5.1f}cm" if front_cm is not None else "F: ---  cm"
    lcd.write_string(front_str.ljust(20))
    lcd.cursor_pos = (3, 0); lcd.write_string(status_line.ljust(20)[:20])

def update_lcd_row(lcd, row, text):
    if lcd is None: return
    lcd.cursor_pos = (row, 0); lcd.write_string(text.ljust(20)[:20])

def log_distances(front_cm, left_cm, right_cm):
    ts = datetime.now().strftime("%H:%M:%S")
    front_str = f"{front_cm:.1f}" if front_cm is not None else "---"
    print(f"[{ts}] F:{front_str}cm L:{left_cm:.1f}cm R:{right_cm:.1f}cm")


# ═══════════════════════════════════════════════════════════════════
#  MOVEMENT HELPERS
# ═══════════════════════════════════════════════════════════════════
def _send_forward(speed=ENTER_SPEED):
    motors.go()
    motors.arduino.write(f"MOVE F {speed} 9999\n".encode())
    motors.wait_for("MOVING", timeout=1)


def move_forward_until_obstacle(laser, lcd=None, threshold_cm=25, speed=ENTER_SPEED):
    _send_forward(speed)
    scan = ydlidar.LaserScan()
    print(f"[move_fwd] Moving until obstacle at {threshold_cm}cm...")
    while True:
        front_cm, left_cm, right_cm = get_stable_distances(laser, scan, num_scans=1)
        if lcd: update_lcd(lcd, left_cm, right_cm, front_cm, "MOVING FWD...")
        if front_cm is not None and front_cm <= threshold_cm:
            motors.stop(); motors.go()
            print(f"[move_fwd] STOPPED at {front_cm:.1f}cm")
            if lcd: update_lcd(lcd, left_cm, right_cm, front_cm, "OBSTACLE")
            return front_cm


def enter_room(side, laser, lcd=None):
    print(f"[enter] Turning {side}...")
    update_lcd_row(lcd, 3, f"TURNING {side}...")
    motors.go()
    motors.turn(side, TURN_DURATION_SEC)


# ═══════════════════════════════════════════════════════════════════
#  PATIENT ASSESSMENT
# ═══════════════════════════════════════════════════════════════════
def start_patient_assessment(lcd=None):
    print("[assessment] Launching client.py...")
    if lcd:
        lcd.clear()
        lcd.cursor_pos = (0, 0); lcd.write_string("PATIENT".ljust(20))
        lcd.cursor_pos = (1, 0); lcd.write_string("ASSESSMENT".ljust(20))
        lcd.cursor_pos = (2, 0); lcd.write_string("IN PROGRESS...".ljust(20))

    result = subprocess.run(["python3", "client.py"], check=False)
    print(f"[assessment] client.py exited {result.returncode}")

    status = "COMPLETE"; flagged = False
    try:
        if os.path.exists(PATIENT_INFO_PATH):
            with open(PATIENT_INFO_PATH, "r") as f:
                all_records = json.load(f)
            if all_records:
                last    = all_records[-1]
                status  = last.get("triage", {}).get("final_status", "COMPLETE")
                flagged = last.get("triage", {}).get("flagged_urgent", False)
    except Exception as e:
        print(f"[assessment] Could not read result: {e}")

    motors.stop()
    print(f"[assessment] Done. Status={status} Urgent={flagged}")

    if lcd:
        while True:
            lcd.clear()
            lcd.cursor_pos = (0, 0); lcd.write_string("== TRIAGE STATUS ==")
            lcd.cursor_pos = (1, 0); lcd.write_string(f"Result: {status}".ljust(20)[:20])
            lcd.cursor_pos = (2, 0)
            lcd.write_string(("!! NURSE ALERTED !!" if flagged else "Assessment Done").ljust(20))
            lcd.cursor_pos = (3, 0); lcd.write_string("Session Complete".ljust(20))
            time.sleep(1)
    else:
        while True:
            print(f"[assessment] {status} | Urgent={flagged}")
            time.sleep(1)


# ═══════════════════════════════════════════════════════════════════
#  PARALLEL BOOT
#  LCD hello animation cut to just clearing + "PULSE READY" — no
#  5-second vanity animation blocking the boot thread.
# ═══════════════════════════════════════════════════════════════════
def _boot_lcd(results, errors):
    try:
        lcd = CharLCD("PCF8574", 0x27, cols=20, rows=4)
        lcd.clear()
        lcd.cursor_pos = (0, 0); lcd.write_string("PULSE".center(20))
        lcd.cursor_pos = (1, 0); lcd.write_string("BOOTING...".center(20))
        results["lcd"] = lcd
        print("[boot] LCD ready")
    except Exception as e:
        errors["lcd"] = e
        print(f"[boot] LCD failed: {e}")


def _boot_arduino(results, errors):
    try:
        motors.connect()
        results["arduino"] = True
        print("[boot] Arduino ready")
    except Exception as e:
        errors["arduino"] = e
        print(f"[boot] Arduino failed: {e}")


def _boot_lidar(results, errors):
    try:
        laser = ydlidar.CYdLidar()
        laser.setlidaropt(ydlidar.LidarPropSerialPort,    LIDAR_PORT)
        laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 128000)
        laser.setlidaropt(ydlidar.LidarPropLidarType,     ydlidar.TYPE_TRIANGLE)
        laser.setlidaropt(ydlidar.LidarPropDeviceType,    ydlidar.YDLIDAR_TYPE_SERIAL)
        laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)
        laser.setlidaropt(ydlidar.LidarPropSampleRate,    5)
        laser.setlidaropt(ydlidar.LidarPropScanFrequency, 12.0)
        if not laser.initialize(): raise RuntimeError("initialize() failed")
        if not laser.turnOn():     raise RuntimeError("turnOn() failed")
        results["laser"] = laser
        print("[boot] LiDAR ready")
    except Exception as e:
        errors["lidar"] = e
        print(f"[boot] LiDAR failed: {e}")


def parallel_boot():
    results = {}; errors = {}
    threads = [
        threading.Thread(target=_boot_lcd,     args=(results, errors), daemon=True),
        threading.Thread(target=_boot_arduino, args=(results, errors), daemon=True),
        threading.Thread(target=_boot_lidar,   args=(results, errors), daemon=True),
    ]
    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    print(f"[boot] Done in {time.time()-t0:.1f}s")

    if "arduino" in errors: raise RuntimeError(f"Arduino: {errors['arduino']}")
    if "lidar"   in errors:
        lcd = results.get("lcd")
        if lcd:
            lcd.clear(); lcd.write_string("LIDAR ERROR".ljust(20))
        raise RuntimeError(f"LiDAR: {errors['lidar']}")

    return results.get("lcd"), results["laser"]


# ═══════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════
def main():
    try:
        lcd, laser = parallel_boot()
    except RuntimeError as e:
        print(f"[main] BOOT FAILED: {e}")
        return

    scan           = ydlidar.LaserScan()
    prev_left_cm   = None
    prev_right_cm  = None
    prev_front_cm  = None
    last_turn_time = 0
    TURN_COOLDOWN  = 3.0
    motor_state    = "IDLE"

    print("[main] Navigation ready.")
    if lcd:
        lcd.clear()
        lcd.cursor_pos = (0, 0); lcd.write_string("PULSE READY".center(20))

    try:
        while True:
            # ── Fast scan: emergency obstacle check ───────────────
            if laser.doProcessSimple(scan):
                front_fast, _, _ = lidarHelpers.get_all_distances(scan, debug=False)

                obstacle_now = (front_fast is not None and front_fast < OBSTACLE_THRESHOLD_CM)
                # Edge case: obstacle was very close and now sensor lost it
                ghost_close  = (front_fast is None and prev_front_cm is not None
                                and prev_front_cm < OBSTACLE_THRESHOLD_CM)

                if obstacle_now or ghost_close:
                    if motor_state != "STOPPED":
                        motors.stop()
                        motor_state = "STOPPED"
                        reason = f"{front_fast:.1f}cm" if front_fast else f"ghost {prev_front_cm:.1f}cm"
                        print(f"[main] STOP — {reason}")

                    front_cm, left_cm, right_cm = lidarHelpers.get_all_distances(scan, debug=False)
                    if front_cm is None: front_cm = prev_front_cm
                    update_lcd(lcd, left_cm or 999, right_cm or 999, front_cm, "OBSTACLE - STOP")

                    # Poll until clear — no sleep, just spin
                    while True:
                        if laser.doProcessSimple(scan):
                            fc, _, _ = lidarHelpers.get_all_distances(scan)
                            if fc is None or fc >= OBSTACLE_THRESHOLD_CM:
                                print("[main] Clear — resuming.")
                                _send_forward()
                                motor_state   = "MOVING"
                                prev_front_cm = fc
                                break
                    continue

                prev_front_cm = front_fast

            # ── Stable scan: room detection + display ─────────────
            front_cm, left_cm, right_cm = get_stable_distances(laser, scan, num_scans=1)
            log_distances(front_cm, left_cm, right_cm)

            now = time.time()

            # Left opening
            if (prev_left_cm is not None and left_cm < 999.0
                    and left_cm - prev_left_cm > OPENING_INCREASE_CM
                    and now - last_turn_time > TURN_COOLDOWN):
                print(f"[OPENING] LEFT — {prev_left_cm:.0f}→{left_cm:.0f}cm")
                motors.stop()
                update_lcd_row(lcd, 3, "OPENING LEFT!")
                enter_room('L', laser, lcd)
                start_patient_assessment(lcd)
                last_turn_time = now; motor_state = "MOVING"
                prev_left_cm = left_cm; prev_right_cm = right_cm

            # Right opening
            elif (prev_right_cm is not None and right_cm < 999.0
                    and right_cm - prev_right_cm > OPENING_INCREASE_CM
                    and now - last_turn_time > TURN_COOLDOWN):
                print(f"[OPENING] RIGHT — {prev_right_cm:.0f}→{right_cm:.0f}cm")
                motors.stop()
                update_lcd_row(lcd, 3, "OPENING RIGHT!")
                enter_room('R', laser, lcd)
                start_patient_assessment(lcd)
                last_turn_time = now; motor_state = "MOVING"
                prev_right_cm = right_cm; prev_left_cm = left_cm

            else:
                update_lcd(lcd, left_cm, right_cm, front_cm, "GOING STRAIGHT")
                if motor_state != "MOVING":
                    _send_forward()
                    motor_state = "MOVING"

            if left_cm  < 999.0: prev_left_cm  = left_cm
            if right_cm < 999.0: prev_right_cm = right_cm

    except KeyboardInterrupt:
        print("[main] Interrupted.")
        motors.stop()
    finally:
        laser.turnOff(); laser.disconnecting()
        if lcd:
            lcd.clear(); lcd.write_string("SYSTEM IDLE".ljust(20))
        motors.go()


if __name__ == "__main__":
    main()