import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import ydlidar
import time
import math
import sys

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

    # --- Setup Visuals ---
    plt.ion()
    fig = plt.figure(figsize=(12, 7), facecolor='black')
    
    # Radar Side
    ax_radar = fig.add_subplot(121, projection='polar', facecolor='#000d00')
    ax_radar.set_ylim(0, 100)
    line, = ax_radar.plot([], [], 'g.', markersize=3, alpha=0.5)
    target_ring, = ax_radar.plot([], [], 'ro', markersize=20, mfc='none', mew=2)
    
    # Text Side
    ax_text = fig.add_subplot(122)
    ax_text.axis('off')
    d_label = ax_text.text(0.5, 0.6, "DIST: --", color='white', fontsize=35, ha='center')
    a_label = ax_text.text(0.5, 0.4, "ANGLE: --", color='lime', fontsize=35, ha='center')

    scan = ydlidar.LaserScan()
    last_ui_update = 0

    try:
        while True:
            if laser.doProcessSimple(scan):
                # PERF: Only update screen 15 times per second
                if time.time() - last_ui_update < 0.06:
                    continue

                angles = []
                distances = []
                closest_d = 999
                closest_a = 0

                for p in scan.points:
                    dist = m_to_in(p.range)
                    if 0.5 < dist < 100.0:
                        angles.append(p.angle)
                        distances.append(dist)
                        if dist < closest_d:
                            closest_d = dist
                            closest_a = p.angle

                # Update Visuals
                line.set_data(angles, distances)
                if closest_d < 999:
                    target_ring.set_data([closest_a], [closest_d])
                    target_ring.set_visible(True)
                    d_label.set_text(f"{closest_d:.1f} in")
                    a_label.set_text(f"{(math.degrees(closest_a)+360)%360:.1f}°")
                else:
                    target_ring.set_visible(False)

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                last_ui_update = time.time()

            time.sleep(0.01) # Keeps the Pi 5 snappy and Ctrl+C working

    except KeyboardInterrupt:
        print("\nKilling Process...")
    finally:
        laser.turnOff()
        laser.disconnecting()
        plt.close('all')
        sys.exit(0) # The final "Exit" hammer

if __name__ == "__main__":
    main()
