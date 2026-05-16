import time
import ydlidar
from RPLCD.i2c import CharLCD
import lcd_hello
import motorControl_Pi as motors
import lidarHelpers
# from modules import vision, tts, patient, stt

# --- Constants ---
LIDAR_PORT              = "/dev/ttyUSB0"
OBSTACLE_THRESHOLD_CM   =025  # Stop threshold for fro5t obstacle
OPENING_INCREASE_2M     = 50  # Increase in side distance to detect opening/doorway


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


def update_lcd_fast(lcd, left_cm, right_cm, front_cm, status_line):
    """Fast LCD update - only updates distances and status, skips header"""
    lcd.cursor_pos = (1, 0)
    lcd.write_string(f"R:{right_cm:5.1f} L:{left_cm:5.1f}cm".ljust(20))
    lcd.cursor_pos = (2, 0)
    front_str = f"F:{front_cm:5.1f}cm" if front_cm is not None else "F: ---  cm"
    lcd.write_string(front_str.ljust(20))
    lcd.cursor_pos = (3, 0)
    lcd.write_string(status_line.ljust(20)[:20])


# ═══════════════════════════════════════════════════════════════════
#  MOTOR CONTROL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
def move_forward_until_obstacle(laser, lcd=None, threshold_cm=25, speed=200):
    """
    Keep moving forward until an obstacle is detected within threshold_cm.
    
    Args:
        laser: Initialized YDLidar object
        lcd: Optional LCD display for real-time distance updates
        threshold_cm: Stop when obstacle closer than this (default: 25cm)
        speed: Motor speed 0-255 (default: 200)
    
    Returns:
        final_distance: Distance to obstacle when stopped (cm)
    """
    motors.go()  # Ensure motors are unlocked
    
    # Start moving forward with long duration (9999 seconds)
    import serial
    cmd = f"MOVE F {speed} 9999\n"
    motors.arduino.write(cmd.encode('utf-8'))
    print(f"[move_forward] Sent: {cmd.strip()}")
    motors.wait_for("MOVING", timeout=3)
    
    scan = ydlidar.LaserScan()
    print(f"[move_forward] Moving forward until obstacle at {threshold_cm}cm...")
    
    while True:
        if laser.doProcessSimple(scan):
            # Get distances from all zones
            front_cm, left_cm, right_cm = lidarHelpers.get_all_distances(scan)
            
            # Update LCD if provided
            if lcd:
                update_lcd(lcd, left_cm, right_cm, front_cm, "MOVING FWD...")
            
            # Check for obstacle
            if front_cm is not None:
                print(f"[move_forward] Front: {front_cm:.1f}cm", end='\r')
                
                if front_cm <= threshold_cm:
                    motors.stop()
                    motors.go()  # Unlock Arduino immediately
                    print(f"\n[move_forward] STOPPED - Obstacle at {front_cm:.1f}cm")
                    
                    if lcd:
                        update_lcd(lcd, left_cm, right_cm, front_cm, "OBSTACLE DETECTED")
                    
                    return front_cm
        
        time.sleep(0.05)


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
    last_turn_time = 0  # Cooldown timer to prevent repeated turns
    TURN_COOLDOWN = 3.0  # seconds
    motor_state = "IDLE"  # Track motor state to avoid repeated commands

    # ═══════════════════════════════════════════════════════════════════
    #  MAIN LOOP — Gallery Navigation with Obstacle Avoidance
    # ═══════════════════════════════════════════════════════════════════
    try:
        while True:
            if not laser.doProcessSimple(scan):
                time.sleep(0.05)
                continue

            # Get distances from all zones in ONE PASS (prevents cross-zone bleeding)
            front_cm, left_cm, right_cm = lidarHelpers.get_all_distances(scan)

            # ═══════════════════════════════════════════════════════════════════
            #  PRIORITY 1: FRONT OBSTACLE — Stop and wait
            # ═══════════════════════════════════════════════════════════════════
            if lidarHelpers.is_obstacle_ahead(scan, threshold_cm=OBSTACLE_THRESHOLD_CM):
                if motor_state != "STOPPED":
                    motors.stop()
                    motor_state = "STOPPED"
                update_lcd(lcd, left_cm, right_cm, front_cm, "OBSTACLE - STOPPED")
                print(f"[main] OBSTACLE ahead — stopped | F:{front_cm} L:{left_cm:.1f} R:{right_cm:.1f}")

                # Wait until obstacle clears
                while True:
                    if laser.doProcessSimple(scan):
                        front_cm, left_cm, right_cm = lidarHelpers.get_all_distances(scan)
                        update_lcd(lcd, left_cm, right_cm, front_cm, "OBSTACLE - STOPPED")
                        
                        if not lidarHelpers.is_obstacle_ahead(scan, threshold_cm=OBSTACLE_THRESHOLD_CM):
                            print("[main] Obstacle cleared — resuming.")
                            motors.go()  # Unlock Arduino before resuming
                            # Send continuous forward movement command
                            cmd = "MOVE F 200 9999\n"
                            motors.arduino.write(cmd.encode('utf-8'))
                            print("[main] Resuming forward motion")
                            motors.wait_for("MOVING", timeout=3)
                            motor_state = "MOVING"
                            break
                    time.sleep(0.05)

            # ═══════════════════════════════════════════════════════════════════
            #  PRIORITY 2: LEFT OPENING — Turn left when opening detected
            # ═══════════════════════════════════════════════════════════════════
            elif (prev_left_cm is not None and 
                  left_cm < 999.0 and 
                  left_cm - prev_left_cm > OPENING_INCREASE_CM and
                  time.time() - last_turn_time > TURN_COOLDOWN):
                
                print(f"[main] Opening on LEFT detected — turning | F:{front_cm} L:{left_cm:.1f} R:{right_cm:.1f}")
                
                # Ensure Arduino is unlocked before turn
                motors.go()
                motors.turn('L', 0.5)
                motor_state = "MOVING"
                
                # Get fresh scan and update LCD
                scan_attempts = 0
                while scan_attempts < 5:
                    if laser.doProcessSimple(scan):
                        front_cm, left_cm, right_cm = lidarHelpers.get_all_distances(scan)
                        break
                    scan_attempts += 1
                    time.sleep(0.01)
                update_lcd(lcd, left_cm, right_cm, front_cm, "GOING STRAIGHT")
                
                # Reset previous values to prevent repeated detection
                prev_left_cm = left_cm if left_cm < 999.0 else prev_left_cm
                prev_right_cm = right_cm if right_cm < 999.0 else prev_right_cm
                last_turn_time = time.time()

            # ═══════════════════════════════════════════════════════════════════
            #  PRIORITY 3: RIGHT OPENING — Turn right when opening detected
            # ═══════════════════════════════════════════════════════════════════
            elif (prev_right_cm is not None and 
                  right_cm < 999.0 and 
                  right_cm - prev_right_cm > OPENING_INCREASE_CM and
                  time.time() - last_turn_time > TURN_COOLDOWN):
                
                print(f"[main] Opening on RIGHT detected — turning | F:{front_cm} L:{left_cm:.1f} R:{right_cm:.1f}")
                
                # Ensure Arduino is unlocked before turn
                motors.go()
                motors.turn('R', 0.5)
                motor_state = "MOVING"
                
                # Get fresh scan and update LCD
                scan_attempts = 0
                while scan_attempts < 5:
                    if laser.doProcessSimple(scan):
                        front_cm, left_cm, right_cm = lidarHelpers.get_all_distances(scan)
                        break
                    scan_attempts += 1
                    time.sleep(0.01)
                update_lcd(lcd, left_cm, right_cm, front_cm, "GOING STRAIGHT")
                
                # Reset previous values to prevent repeated detection
                prev_left_cm = left_cm if left_cm < 999.0 else prev_left_cm
                prev_right_cm = right_cm if right_cm < 999.0 else prev_right_cm
                last_turn_time = time.time()

            # ═══════════════════════════════════════════════════════════════════
            #  PRIORITY 4: DEFAULT — Go straight (no obstacles or openings)
            # ═══════════════════════════════════════════════════════════════════
            else:
                update_lcd(lcd, left_cm, right_cm, front_cm, "GOING STRAIGHT")
                if motor_state != "MOVING":
                    motors.go()
                    # Send continuous forward movement command (9999 seconds = continuous)
                    cmd = "MOVE F 200 9999\n"
                    motors.arduino.write(cmd.encode('utf-8'))
                    print("[main] Starting forward motion")
                    motors.wait_for("MOVING", timeout=3)
                    motor_state = "MOVING"

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