import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import ydlidar
import time
import math

# Prototype 3.1 - Pulse "Wide-Eye" Radar
def m_to_in(meters):
    return meters * 39.3701

def main():
    laser = ydlidar.CYdLidar()
    laser.setlidaropt(ydlidar.LidarPropSerialPort, "/dev/ttyUSB0")
    laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 128000)
    laser.setlidaropt(ydlidar.LidarPropLidarType, 18) 
    laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)

    if not laser.initialize(): return
    laser.turnOn()

    # --- BIGGER MAP SETUP ---
    plt.ion()
    # figsize=(12, 10) makes it fill more of a widescreen monitor
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': 'polar'})
    fig.canvas.manager.set_window_title('Pulse Lidar 100-Inch Radar')
    
    fig.patch.set_facecolor('black')
    ax.set_facecolor('#000d00') # Darker green background
    
    # Grid and Ring setup
    ax.set_ylim(0, 100) # THE BIG 100 INCH LIMIT
    ax.set_yticks([25, 50, 75, 100]) # Distance rings
    ax.set_yticklabels(['25"', '50"', '75"', '100"'], color='green', fontsize=10)
    ax.grid(color='#004400', linestyle='-', alpha=0.7)
    ax.tick_params(axis='x', colors='green') # Degree markers
    
    # BIGGER DOTS: markersize=5 makes hand/face detections much clearer
    line, = ax.plot([], [], 'g.', markersize=5) 
    
    scan = ydlidar.LaserScan()

    try:
        while True:
            if laser.doProcessSimple(scan):
                angles = []
                distances = []

                for point in scan.points:
                    dist_in = m_to_in(point.range)
                    # We keep radians for the math, but filter for 100 inches
                    if 0.5 < dist_in < 100.0:
                        angles.append(point.angle)
                        distances.append(dist_in)

                line.set_data(angles, distances)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nRadar Powered Down.")
    finally:
        laser.turnOff()
        laser.disconnecting()
        plt.close()

if __name__ == "__main__":
    main()
