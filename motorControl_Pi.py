import serial
import time

# Connect to Arduino
try:
    arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1) #Change to whereever the Arduino is connected
    time.sleep(2)  # Wait for Arduino to boot
    print(arduino.readline().decode().strip())
except serial.SerialException:
    print("Error: Could not connect to Arduino on /dev/ttyACM0")
    print("Check that the Arduino is plugged in and the port is correct")
    exit()

def wait_for(response, timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        line = arduino.readline().decode().strip()
        if line:
            print(f"Arduino: {line}")
        if line == response:
            return True
    return False

# Move function with built-in pauses
# Syntax: move(direction, speed, seconds) e.g. move('F', 200, 1)
def move(direction, speed, seconds):
    cmd = f"MOVE {direction} {speed} {seconds}\n"  # Create command
    arduino.write(cmd.encode('utf-8'))             # Send to Arduino
    print(f"Sent: {cmd.strip()}")                 # Debug print
    wait_for("DONE", timeout=seconds + 10)        # Wait for Arduino to confirm done

# Turn function with built-in pauses
# Syntax: turn(side, seconds) e.g. turn('L', 1)
def turn(side, seconds):
    cmd = f"TURN {side} {seconds}\n"
    arduino.write(cmd.encode('utf-8'))
    print(f"Sent: {cmd.strip()}")
    wait_for("DONE", timeout=seconds + 10)        # Wait for Arduino to confirm done

def stop():
    arduino.write(b"STOP\n")                      # Send to Arduino
    print("Sent: STOP")                           # Debug print
    wait_for("STOPPED", timeout=3)

def go():
    arduino.write(b"GO\n")                        # Send to Arduino
    print("Sent: GO")                             # Debug print
    wait_for("UNLOCKED", timeout=3)

if __name__ == "__main__":
    try:
        move('F', 200, 1)  # Forward 1 second
        move('B', 200, 1)  # Backward 1 second
        turn('L', 1)       # Left turn 1 second
        turn('R', 1)       # Right turn 1 second
    finally:
        arduino.close()