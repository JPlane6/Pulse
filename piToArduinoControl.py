import serial
import time

ser = serial.Serial('/dev/ttyACM0', 9600)
time.sleep(2)  # let Arduino reset

def forward():
    ser.write(b'f\n')

def backward():
    ser.write(b'b\n')

def stop():
    ser.write(b's\n')

# Test motor
forward()
time.sleep(2)
stop()
time.sleep(1)
backward()
time.sleep(2)
stop()
