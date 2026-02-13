import RPi.GPIO as GPIO
import time

class DKMotorShield:
    """
    Modular L293D + 74HC595 motor library for Raspberry Pi.
    Controls 4 motors via shift register with PWM-like speed control.
    """

    # Motor bit mapping
    MOTOR_BITS = {
        1: {'F': 0b00000001, 'B': 0b00000010},
        2: {'F': 0b00000100, 'B': 0b00001000},
        3: {'F': 0b00010000, 'B': 0b00100000},
        4: {'F': 0b01000000, 'B': 0b10000000},
    }

    def __init__(self, data_pin, clock_pin, latch_pin, enable_pin):
        self.DATA = data_pin
        self.CLOCK = clock_pin
        self.LATCH = latch_pin
        self.ENABLE = enable_pin

        # GPIO setup
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.DATA, GPIO.OUT)
        GPIO.setup(self.CLOCK, GPIO.OUT)
        GPIO.setup(self.LATCH, GPIO.OUT)
        GPIO.setup(self.ENABLE, GPIO.OUT)
        GPIO.output(self.ENABLE, GPIO.HIGH)

    def _shift_out(self, byte):
        GPIO.output(self.LATCH, 0)
        for i in range(8):
            bit = (byte >> (7 - i)) & 1
            GPIO.output(self.DATA, bit)
            GPIO.output(self.CLOCK, 1)
            GPIO.output(self.CLOCK, 0)
        GPIO.output(self.LATCH, 1)

    def _scale_speed(self, bits, speed=100, steps=20, delay=0.01):
        """
        Ramp motors up/down for smoother start/stop.
        bits: 8-bit pattern for motors
        speed: 0-100%
        steps: number of increments for scaling
        delay: delay between steps
        """
        scaled_steps = int(speed / 100 * steps)
        for i in range(1, scaled_steps + 1):
            duty = int(bits * i / steps)
            self._shift_out(duty)
            time.sleep(delay)

    # Movement functions with optional speed
    def forward(self, motors=[1,2,3,4], speed=100):
        bits = 0
        for m in motors:
            bits |= self.MOTOR_BITS[m]['F']
        self._scale_speed(bits, speed=speed)

    def backward(self, motors=[1,2,3,4], speed=100):
        bits = 0
        for m in motors:
            bits |= self.MOTOR_BITS[m]['B']
        self._scale_speed(bits, speed=speed)

    def stop(self):
        self._shift_out(0)