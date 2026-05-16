import ydlidar, math, time

laser = ydlidar.CYdLidar()
laser.setlidaropt(ydlidar.LidarPropSerialPort, '/dev/ttyUSB0')
laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 128000)
laser.setlidaropt(ydlidar.LidarPropLidarType, 18)
laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)
laser.initialize()
laser.turnOn()
scan = ydlidar.LaserScan()
time.sleep(2)
print('Put hand on RIGHT side close (~15cm) and press Enter')
input()
laser.doProcessSimple(scan)
points = [(math.degrees(p.angle) % 360, p.range*100) for p in scan.points if 3 < p.range*100 < 25]
for angle, dist in sorted(points):
    print(f'angle: {angle:.1f} dist: {dist:.1f}cm')
laser.turnOff()
laser.disconnecting()