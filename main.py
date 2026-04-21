import time
import sys
import ydlidar
from driver import RobotDriver
from RPLCD.i2c import CharLCD

# --- HARDWARE MAPPING ---
# Arduino is usually ACM0, LiDAR is usually USB0
# If LiDAR doesn't appear, try changing "/dev/ttyUSB0" to "/dev/ttyACM1"
LIDAR_PORT = "/dev/ttyUSB0" 
ARDUINO_PORT = "/dev/ttyACM0"

# --- INITIALIZE HARDWARE ---
print("Step 1: Connecting to Muscles (Arduino)...")
robot = RobotDriver(port=ARDUINO_PORT)

print("Step 2: Connecting to Eyes (LiDAR)...")
laser = ydlidar.CYdLidar()
laser.setlidaropt(ydlidar.LidarPropSerialPort, LIDAR_PORT)
laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 128000)
laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)

# Power-saving / Stability settings for Pi 5
laser.setlidaropt(ydlidar.LidarPropSampleRate, 3) 
laser.setlidaropt(ydlidar.LidarPropScanFrequency, 5.0) 

print("Step 3: Connecting to Screen (LCD)...")
try:
    lcd = CharLCD('PCF8574', 0x27, cols=20, rows=4)
except:
    print("LCD not found, skipping display.")
    lcd = None

def main():
    if lcd:
        lcd.clear()
        lcd.write_string("PULSE SYSTEM\nINITIALIZING...")
    
    # Wait for power to stabilize
    time.sleep(2) 

    if not laser.initialize():
        print(f"ERROR: Cannot find LiDAR on {LIDAR_PORT}. check cables!")
        return

    laser.turnOn()
    print("System Online. Starting Loop.")
    
    scan = ydlidar.LaserScan()

    try:
        while True:
            if laser.doProcessSimple(scan):
                # Filter points in a 60-degree cone in front (330 to 30 deg)
                # Angle in Radians: > 5.75 or < 0.52
                front_points = [p.range for p in scan.points if (p.angle > 5.75 or p.angle < 0.52) and p.range > 0.1]
                
                dist = min(front_points) * 100 if front_points else 999
                
                if lcd:
                    lcd.cursor_pos = (2, 0)
                    lcd.write_string(f"Dist: {dist:>5.1f} cm   ")

                # --- DECISION LOGIC ---
                if dist < 45:
                    if lcd:
                        lcd.cursor_pos = (3, 0)
                        lcd.write_string("STATUS: TURN LEFT  ")
                    robot.stop()
                    time.sleep(0.2)
                    robot.turn('L', 1) # Command: TURN L 1
                else:
                    if lcd:
                        lcd.cursor_pos = (3, 0)
                        lcd.write_string("STATUS: MOVING FWD ")
                    robot.move_forward(speed=170, duration=0.3)
            
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        laser.turnOff()
        laser.disconnecting()
        robot.stop()
        if lcd:
            lcd.clear()
            lcd.write_string("SYSTEM IDLE")

if __name__ == "__main__":
    main()