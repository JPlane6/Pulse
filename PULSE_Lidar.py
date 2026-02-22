#Prototype 1 - Pulse YDLiDAR



import ydlidar

import time

import math



# Helper function to convert meters to inches

def m_to_in(meters):

    return meters * 39.3701



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

    print("Monitoring distance... Press Ctrl+C to stop.")



    try:

        while True: # The infinite loop

            if laser.doProcessSimple(scan):

                # We want to check every point in a 360-degree scan

                for point in scan.points:

                    dist_inches = m_to_in(point.range)

                    angle = math.degrees(point.angle) # Convert angle from radians to degrees

                    # Ignore 0.0 (error/no return) and check if within 5 inches

                    if 0.1 < dist_inches <= 15.0:

                        print(f"OBJECT DETECTED! Distance: {dist_inches:.2f} inches, at a {point.angle:.2f}° angle")

                        # Add a tiny sleep or break if you don't want 1000 prints per second

                        break

           

            time.sleep(0.05) # Small delay to prevent CPU overload



    except KeyboardInterrupt:

        print("\nStopping...")

    finally:

        laser.turnOff()

        laser.disconnecting()



if __name__ == "__main__":

    main()