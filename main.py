from dk_motor_shield import DKMotorShield
import time

robot = DKMotorShield(
    data_pin=27,   # DATA (D12)
    clock_pin=22,  # CLOCK (D4)
    latch_pin=17,  # LATCH (D8)
    enable_pin=18  # ENABLE (D7)
)

try:
    print("Forward at 50% speed")
    robot.forward(speed=50)
    time.sleep(2)

    print("Backward at 80% speed")
    robot.backward(speed=80)
    time.sleep(2)

finally:
    print("Stopping motors")
    robot.stop()
