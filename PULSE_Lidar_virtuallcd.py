# Prototype 2 - Pulse YDLiDAR (Virtual LCD Edition)

import ydlidar
import time
import math

def m_to_in(meters):
    return meters * 39.3701

def get_side(angle):
    """Determines the side based on angle (0 is front)"""
    if angle > 315 or angle <= 45:
        return "FRONT"
    elif 45 < angle <= 135:
        return "LEFT"
    elif 135 < angle <= 225:
        return "BACK"
    elif 225 < angle <= 315:
        return "RIGHT"
    return "UNKNOWN"

def virtual_lcd(line1, line2):
    """Simulates a 16x2 LCD screen in the terminal"""
    # This clears the terminal line so it looks like a screen update

    '''
    \033[H moves the cursor to the top-left of the terminal.
    \033[J clears everything from the cursor to the end of the screen.
    This prevents the terminal from scrolling down; it makes the text
    stay in one spot.
    '''
    print("\033[H\033[J", end="") 

    # This prints the top border of our "fake" LCD screen.
    print(" ------------------ ")
    

    '''
    .center(16) takes your string (like "I see an object") and adds 
    spaces on the left and right until it is exactly 16 characters
    long (the width of our LCD). This way, the text is nicely centered on the screen.
    The '|' symbols create the side borders of the screen.
    
    '''
    print(f"| {line1.center(16)} |")
    print(f"| {line2.center(16)} |")
    print(" ------------------ ")

    
    print("\n(Press Ctrl+C to stop)")

def main():
    laser = ydlidar.CYdLidar()

    # --- Hardware Setup ---

    laser.setlidaropt(ydlidar.LidarPropSerialPort, "/dev/ttyUSB0")
    laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 128000)
    laser.setlidaropt(ydlidar.LidarPropLidarType, 18) # Using the 18 'fix' we found
    laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)

    # Initialize
    if not laser.initialize():
        print("Failed to initialize LiDAR")
        return

    if not laser.turnOn():
        print("Failed to start motor")
        return

    scan = ydlidar.LaserScan()

    last_side = "STARTUP"
    
    print("Monitoring distance... Press Ctrl+C to stop.")

    try:
        while True: # The infinite loop
            if laser.doProcessSimple(scan):
                # We want to check every point in a 360-degree scan
                closest_dist = 999 # Resets closest distance for every new rotation
                closest_side = ""  # Resets the found side for every new rotation
                
                for point in scan.points:
                    dist_inches = m_to_in(point.range) # Convert current point to inches
                    
                    # Normalizes angle to a 0-360 range for easier logic
                    angle = (math.degrees(point.angle) + 360) % 360

                    # Filters for objects within 10 inches but outside sensor body
                    if 0.5 < dist_inches <= 10.0:
                        if dist_inches < closest_dist: # Checks if this point is the nearest one yet
                            closest_dist = dist_inches # Updates nearest distance
                            closest_side = get_side(angle) # Identifies which side it's on
                        


                    
                    if closest_side != last_side:
                        if closest_side:
                            virtual_lcd("OBJECT DETECTED", f"ON MY {closest_side}")
                        else:
                            virtual_lcd("PATH CLEAR", "SCANNING...")
                        last_side = closest_side # Saves current side to memory   
            time.sleep(0.05) # Small delay to prevent CPU overload

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        laser.turnOff()
        laser.disconnecting()

if __name__ == "__main__":
    main()