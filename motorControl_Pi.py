import serial
import time

# Connect to Arduino
arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1) #Change to whereever the Arduino is connected
time.sleep(2)  # Wait for Arduino to boot

# Move function with built-in pauses
def move(direction, speed, seconds):
    cmd = f"MOVE {direction} {speed} {seconds}\n"  # Create command
    arduino.write(cmd.encode('utf-8'))             # Send to Arduino
    print(f"Sent: {cmd.strip()}")                 # Debug print
    time.sleep(0.05)  # 50ms pause for Arduino to process command
    time.sleep(seconds)  # Wait while motors run
    time.sleep(0.1)  # 100ms extra leeway after running

# Turn function with built-in pauses
def turn(side, seconds):
    cmd = f"TURN {side} {seconds}\n"
    arduino.write(cmd.encode('utf-8'))
    print(f"Sent: {cmd.strip()}")
    time.sleep(0.05)   # 50ms pause for Arduino to process command
    time.sleep(seconds)  # Wait while turning
    time.sleep(0.1)  # 100ms leeway after turn

if __name__ == "__main__":
    move('F', 200, 1)  # Forward 1 second
    move('B', 200, 1)  # Backward 1 second
    turn('L', 1)       # Left turn 1 second
    turn('R', 1)       # Right turn 1 second