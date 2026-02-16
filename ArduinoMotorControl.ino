#include "AFMotor_R4.h" // Import Adafruit Motor Driver Library

AF_DCMotor motor1(1);  // Create motor on M1
AF_DCMotor motor2(2);  // Create motor on M2
AF_DCMotor motor3(3);  // Create motor on M3
AF_DCMotor motor4(4);  // Create motor on M4

//-------------------------------------|
// Protoype Methods
void move(uint8_t direction ,uint16_t seconds, uint8_t speed, AF_DCMotor &motor);
//-------------------------------------|


void setup() {

//-------------------------------------|
// Set initial speed and Brake Motors
motor1.setSpeed(0);
motor1.run(BRAKE);

motor2.setSpeed(0);
motor2.run(BRAKE);

motor3.setSpeed(0);
motor3.run(BRAKE);

motor4.setSpeed(0);
motor4.run(BRAKE);
//-------------------------------------|

}

void loop() {
  
  move(FORWARD, 1, 200, motor1);  // Test the function
  move(BACKWARD, 1, 200, motor1);
  while(true){} // Park the Robot here after Test runs

}
// Method for smoothly moving forward for parameters seconds, speed, and which motor is affected
void move(uint8_t direction ,uint16_t seconds, uint8_t speed, AF_DCMotor &motor) {

  motor.run(direction);

  // Smoothly scale the motor up to speed specified
  for (uint8_t i = 0; i < speed; i++) {
    motor.setSpeed(i);
    delay(10);
  }

  // To avoid an infinite loop due to i being an unsigned integer(they loop around) make max speed manually 255 if needed
  if (speed == 255) {
    motor.setSpeed(255);
  }

  delay(seconds * 1000);  // let the motor run for a specified amount of time(delay() parameter is in milliseconds)

  // Smoothly scale down the motor to stop it
  for (uint8_t i = speed; i > 0; i--) {
    motor.setSpeed(i);
    delay(10);
  }

  motor.setSpeed(0);  // To avoid an infinite loop manually set speed to 0
  
  motor.run(RELEASE); // Cut power to motor after it is done running
  delay(10);

}