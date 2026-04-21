import serial
import time

class RobotDriver:
    def __init__(self, port='/dev/ttyACM0', baud=9600):
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            # Give Arduino time to reset after the serial connection opens
            time.sleep(2) 
            print(f"SUCCESS: Arduino ready on {port}")
        except Exception as e:
            print(f"FAIL: Could not open Arduino port: {e}")
            self.ser = None

    def send(self, cmd):
        if self.ser:
            try:
                self.ser.write(f"{cmd}\n".encode())
            except:
                print("Error: Serial write failed.")

    def move_forward(self, speed=160, duration=0.5):
        self.send("GO") # Ensure safety lock is released
        self.send(f"MOVE F {speed} {duration}")

    def stop(self):
        self.send("STOP")

    def turn(self, side, duration=1):
        self.send("GO")
        # Command format for your Arduino: TURN [L/R] [SECONDS]
        self.send(f"TURN {side} {duration}")