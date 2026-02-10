#THIS CODE ONLY WORKS WHEN SSHed INTO THE RASPBERRY PI
import os
import ydlidar
import time

# 1. Basic Print Test
print("--- SCRIPT STARTED ---")

# 2. Setup the Lidar
laser = ydlidar.CYdLidar()

# Set the Port (Check if yours is USB0 or USB1)
port = "/dev/ttyUSB0"
laser.setlidaropt(ydlidar.LidarPropSerialPort, port)

# Set the Baudrate (Try 128000 first, then 115200 or 12800)
laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 115200)

# Set the Device Type (0 = G2, 1 = F4, 15 = X4, 18 = X2/X2L)
laser.setlidaropt(ydlidar.LidarPropLidarType, 18)
print(f"Connecting to {port}...")

# 3. Initialize
ret = laser.initialize()

if ret:
    print("Connection Successful! Starting Motor...")
    ret = laser.turnOn()
    if ret:
        print("Motor is spinning. Fetching data...")
        scan = ydlidar.LaserScan()
        
        # Take 5 readings then stop
        for i in range(5):
            if laser.doProcessSimple(scan):
                print(f"Scan {i+1}: I see {scan.points.size()} points.")
            time.sleep(0.5)
    else:
        print("Failed to start motor.")
else:
    print("Failed to initialize. Check Baudrate or Port.")

# 4. Clean up
laser.turnOff()
laser.disconnecting()
print("--- SCRIPT FINISHED ---")