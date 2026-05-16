#!/usr/bin/env python3
"""
Hospital Gallery Navigation Robot - SIMPLIFIED
===============================================
CURRENT MODE: Front Obstacle Detection Only

Environment: Narrow corridor
Behavior:
  - Move forward continuously
  - STOP when wall/obstacle detected ahead
  - Resume when obstacle clears
  
NOTE: Left/Right room detection currently DISABLED for testing
"""

import time
import ydlidar
from RPLCD.i2c import CharLCD
import motorControl_Pi as motors
import lidarHelpers

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
LIDAR_PORT = "/dev/ttyUSB0"

# Detection thresholds
FRONT_WALL_THRESHOLD_CM = 70      # Stop if front wall closer than this
CORRIDOR_WIDTH_CM = 40             # Normal corridor width (both sides)
OPENING_INCREASE_CM = 50           # Room opening = side distance increases by this much

# Turn parameters
TURN_DURATION_SEC = 0.5            # How long to turn (adjust based on robot)
TURN_COOLDOWN_SEC = 3.0            # Prevent repeated turn detection

# Motor parameters
FORWARD_SPEED = 200                # Motor speed for forward movement (0-255)


# ═══════════════════════════════════════════════════════════════════
#  LCD DISPLAY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
def init_lcd():
    """Initialize 20x4 LCD display."""
    try:
        lcd = CharLCD("PCF8574", 0x27, cols=20, rows=4)
        lcd.clear()
        return lcd
    except Exception as e:
        print(f"[ERROR] Could not connect to LCD: {e}")
        return None


def display_status(lcd, front_cm, left_cm, right_cm, status_msg):
    """
    Update LCD with current distances and status.
    Row 0: Header
    Row 1: Left & Right distances
    Row 2: Front distance
    Row 3: Status message
    """
    if lcd is None:
        return
    
    lcd.cursor_pos = (0, 0)
    lcd.write_string("GALLERY ROBOT".ljust(20))
    
    lcd.cursor_pos = (1, 0)
    left_str = f"{left_cm:.0f}" if left_cm < 999 else "---"
    right_str = f"{right_cm:.0f}" if right_cm < 999 else "---"
    lcd.write_string(f"L:{left_str:>4} R:{right_str:>4}cm".ljust(20))
    
    lcd.cursor_pos = (2, 0)
    front_str = f"{front_cm:.0f}" if front_cm is not None else "---"
    lcd.write_string(f"Front: {front_str:>4}cm".ljust(20))
    
    lcd.cursor_pos = (3, 0)
    lcd.write_string(status_msg.ljust(20)[:20])


# ═══════════════════════════════════════════════════════════════════
#  LIDAR SENSOR FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
def init_lidar_sensor(lcd):
    """Initialize YDLidar X4 sensor."""
    laser = ydlidar.CYdLidar()
    laser.setlidaropt(ydlidar.LidarPropSerialPort, LIDAR_PORT)
    laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 128000)
    laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
    laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
    laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)
    laser.setlidaropt(ydlidar.LidarPropSampleRate, 5)
    laser.setlidaropt(ydlidar.LidarPropScanFrequency, 12.0)
    
    if not laser.initialize():
        print("[ERROR] Failed to initialize LiDAR")
        if lcd:
            display_status(lcd, None, 999, 999, "LIDAR INIT ERROR")
        return None
    
    if not laser.turnOn():
        print("[ERROR] Failed to start LiDAR motor")
        if lcd:
            display_status(lcd, None, 999, 999, "LIDAR MOTOR ERROR")
        return None
    
    print("[OK] LiDAR initialized successfully")
    return laser


def get_distances(laser):
    """
    Get single-scan distances from all three zones.
    Returns: (front_cm, left_cm, right_cm)
      - front_cm: Distance ahead (None if no obstacle)
      - left_cm: Left wall distance (999.0 if no wall)
      - right_cm: Right wall distance (999.0 if no wall)
    """
    scan = ydlidar.LaserScan()
    
    if laser.doProcessSimple(scan):
        return lidarHelpers.get_all_distances(scan, debug=True)  # Enable debug output
    else:
        return None, 999.0, 999.0


# ═══════════════════════════════════════════════════════════════════
#  DECISION LOGIC
# ═══════════════════════════════════════════════════════════════════
def detect_front_wall(front_cm):
    """Check if there's a wall ahead."""
    if front_cm is None:
        return False
    is_wall = front_cm < FRONT_WALL_THRESHOLD_CM
    
    # Debug output
    if is_wall:
        print(f"[DETECT] Front wall detected: {front_cm:.1f}cm < {FRONT_WALL_THRESHOLD_CM}cm threshold")
    
    return is_wall


def detect_left_opening(left_cm, prev_left_cm):
    """
    Detect room opening on left side.
    Opening = left distance suddenly increases (wall ends, room begins)
    """
    if prev_left_cm is None or prev_left_cm >= 999.0:
        return False
    if left_cm >= 999.0:
        return False
    
    distance_increase = left_cm - prev_left_cm
    return distance_increase > OPENING_INCREASE_CM


def detect_right_opening(right_cm, prev_right_cm):
    """
    Detect room opening on right side.
    Opening = right distance suddenly increases (wall ends, room begins)
    """
    if prev_right_cm is None or prev_right_cm >= 999.0:
        return False
    if right_cm >= 999.0:
        return False
    
    distance_increase = right_cm - prev_right_cm
    return distance_increase > OPENING_INCREASE_CM


# ═══════════════════════════════════════════════════════════════════
#  MOTOR ACTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
def action_stop(lcd, front_cm, left_cm, right_cm, reason="STOPPED"):
    """Stop the robot."""
    motors.stop()
    display_status(lcd, front_cm, left_cm, right_cm, reason)
    print(f"[ACTION] STOP - {reason} | F:{front_cm} L:{left_cm:.0f} R:{right_cm:.0f}")


def action_move_forward(lcd, front_cm, left_cm, right_cm):
    """Start moving forward."""
    motors.go()
    cmd = "MOVE F 200 9999\n"  # Continuous forward
    motors.arduino.write(cmd.encode('utf-8'))
    motors.wait_for("MOVING", timeout=3)
    display_status(lcd, front_cm, left_cm, right_cm, "MOVING FORWARD")
    print(f"[ACTION] FORWARD | F:{front_cm} L:{left_cm:.0f} R:{right_cm:.0f}")


def action_turn_left(lcd, front_cm, left_cm, right_cm):
    """Turn left and enter room."""
    motors.stop()
    time.sleep(0.2)
    display_status(lcd, front_cm, left_cm, right_cm, "ROOM LEFT - TURNING")
    print(f"[ACTION] TURN LEFT - Room detected | L:{left_cm:.0f}")
    
    motors.go()
    motors.turn('L', TURN_DURATION_SEC)
    
    display_status(lcd, front_cm, left_cm, right_cm, "TURNED LEFT")
    print("[ACTION] Turn complete")


def action_turn_right(lcd, front_cm, left_cm, right_cm):
    """Turn right and enter room."""
    motors.stop()
    time.sleep(0.2)
    display_status(lcd, front_cm, left_cm, right_cm, "ROOM RIGHT - TURNING")
    print(f"[ACTION] TURN RIGHT - Room detected | R:{right_cm:.0f}")
    
    motors.go()
    motors.turn('R', TURN_DURATION_SEC)
    
    display_status(lcd, front_cm, left_cm, right_cm, "TURNED RIGHT")
    print("[ACTION] Turn complete")


# ═══════════════════════════════════════════════════════════════════
#  MAIN CONTROL LOOP
# ═══════════════════════════════════════════════════════════════════
def main():
    """Main robot control loop."""
    
    # ─────────────────────────────────────────────────────────────
    # INITIALIZATION
    # ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("HOSPITAL GALLERY NAVIGATION ROBOT - SIMPLIFIED MODE")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Front wall threshold: {FRONT_WALL_THRESHOLD_CM}cm")
    print(f"  - Front detection zone: 53-73° (63±10°)")
    print(f"  - Minimum detection distance: 35cm (ignores robot chassis)")
    print(f"  - Left/Right detection: DISABLED")
    print(f"  - Motor speed: {FORWARD_SPEED}")
    print(f"  - Debug angle output: ENABLED")
    print("=" * 60)
    
    lcd = init_lcd()
    if lcd:
        display_status(lcd, None, 999, 999, "INITIALIZING...")
    
    laser = init_lidar_sensor(lcd)
    if laser is None:
        return
    
    # Robot state
    robot_moving = False
    
    if lcd:
        display_status(lcd, None, 999, 999, "READY")
    
    print("\n[READY] Starting navigation...")
    print("[INFO] Left/Right detection DISABLED - Front obstacle detection only")
    time.sleep(1)
    
    # ─────────────────────────────────────────────────────────────
    # MAIN CONTROL LOOP
    # ─────────────────────────────────────────────────────────────
    try:
        while True:
            # ┌─────────────────────────────────────────────────┐
            # │ STEP 1: GET SENSOR DATA                         │
            # └─────────────────────────────────────────────────┘
            front_cm, left_cm, right_cm = get_distances(laser)
            
            # Print distances for debugging
            front_str = f"{front_cm:.1f}" if front_cm is not None else "---"
            print(f"[SENSOR] Front: {front_str}cm | Left: {left_cm:.1f}cm | Right: {right_cm:.1f}cm")
            
            # ┌─────────────────────────────────────────────────┐
            # │ STEP 2: DECISION LOGIC (FRONT ONLY)             │
            # └─────────────────────────────────────────────────┘
            
            # FRONT WALL DETECTED → STOP
            # ───────────────────────────────────────────────────
            if detect_front_wall(front_cm):
                if robot_moving:
                    action_stop(lcd, front_cm, left_cm, right_cm, "WALL AHEAD - STOP")
                    robot_moving = False
                    print(f"[DECISION] Front wall at {front_cm:.1f}cm - STOPPING")
                else:
                    # Already stopped, just update display
                    display_status(lcd, front_cm, left_cm, right_cm, "WALL AHEAD - STOP")
                
                # Wait for wall to clear
                while detect_front_wall(front_cm):
                    time.sleep(0.1)
                    front_cm, left_cm, right_cm = get_distances(laser)
                    display_status(lcd, front_cm, left_cm, right_cm, "WAITING...")
                
                print("[INFO] Wall cleared, resuming")
                continue
            
            # NO FRONT WALL → MOVE FORWARD
            # ───────────────────────────────────────────────────
            else:
                if not robot_moving:
                    action_move_forward(lcd, front_cm, left_cm, right_cm)
                    robot_moving = True
                    print(f"[DECISION] Path clear - MOVING FORWARD")
                else:
                    # Already moving, just update display
                    display_status(lcd, front_cm, left_cm, right_cm, "MOVING FORWARD")
            
            # Small delay for loop timing
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")
    
    finally:
        # ─────────────────────────────────────────────────────────────
        # CLEANUP
        # ─────────────────────────────────────────────────────────────
        print("[INFO] Shutting down...")
        motors.stop()
        laser.turnOff()
        laser.disconnecting()
        
        if lcd:
            lcd.clear()
            lcd.write_string("SYSTEM STOPPED".ljust(20))
        
        print("[INFO] Shutdown complete")


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
