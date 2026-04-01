import serial
import time

# Connect to Arduino
try:
    arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)  # Update port if needed
    time.sleep(2)  # Wait for Arduino to boot
    print(arduino.readline().decode().strip())
except serial.SerialException:
    print("Error: Could not connect to Arduino on /dev/ttyACM0")
    print("Check that the Arduino is plugged in and the port is correct")
    exit()

def read_response(timeout=0.1):
    start = time.time()
    while time.time() - start < timeout:
        if arduino.in_waiting:
            line = arduino.readline().decode().strip()
            if line:
                print(f"ARDUINO: {line}")

def send_command(cmd):
    """
    Send a command string to Arduino and read responses until it's done.
    """
    arduino.write(f"{cmd}\n".encode('utf-8'))
    print(f"SENT: {cmd}")
    time.sleep(0.05)  # Small delay to allow Arduino to respond
    read_response(timeout=0.5)

def serial_terminal():
    """
    Main loop: read user input, send to Arduino, display responses.
    Ends when user types SerialEND.
    """
    print("Serial terminal started. Type commands to send. Type 'SerialEND' to quit.")
    try:
        while True:
            cmd = input(">>> ").strip()
            if cmd == "SerialEND":
                print("Exiting serial terminal...")
                break
            if cmd:
                send_command(cmd)
                # Read any additional lines for a short time
                read_response(timeout=1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        arduino.close()
        print("Serial connection closed.")

if __name__ == "__main__":
    serial_terminal()