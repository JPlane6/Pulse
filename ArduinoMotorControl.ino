#include "AFMotor_R4.h" // Import Adafruit Motor Driver Library

AF_DCMotor motor1(1);  // Create motor on M1
AF_DCMotor motor2(2);  // Create motor on M2
AF_DCMotor motor3(3);  // Create motor on M3
AF_DCMotor motor4(4);  // Create motor on M4

//-------------------------------------|
// Protoype Methods
void moveAll(uint8_t direction ,uint16_t seconds, uint8_t speed);
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
  
  //-------------------------------------|
  // Test the function
  moveAll(FORWARD, 1, 200);
  moveAll(BACKWARD, 1, 200);
  //-------------------------------------|
  
  while(true){} // Park the Robot here after Test runs

}


// Method for smoothly moving forward for parameters seconds, speed, and all motors together
void moveAll(uint8_t direction ,uint16_t seconds, uint8_t speed) {

  // Set all motors to the chosen direction
  motor1.run(direction);
  motor2.run(direction);
  motor3.run(direction);
  motor4.run(direction);
  //-------------------------------------|

  // Smoothly scale the motors up to speed specified
  for (uint8_t i = 0; i < speed; i++) {
    motor1.setSpeed(i);
    motor2.setSpeed(i);
    motor3.setSpeed(i);
    motor4.setSpeed(i);
    delay(10);
  }

  // To avoid an infinite loop due to i being an unsigned integer(they loop around) make max speed manually 255 if needed
  if (speed == 255) {
    motor1.setSpeed(255);
    motor2.setSpeed(255);
    motor3.setSpeed(255);
    motor4.setSpeed(255);
  }

  delay(seconds * 1000);  // let the motors run for a specified amount of time(delay() parameter is in milliseconds)

  // Smoothly scale down the motors to stop them
  for (uint8_t i = speed; i > 0; i--) {
    motor1.setSpeed(i);
    motor2.setSpeed(i);
    motor3.setSpeed(i);
    motor4.setSpeed(i);
    delay(10);
  }

  
  // To avoid an infinite loop manually set speed to 0
  motor1.setSpeed(0);
  motor2.setSpeed(0);
  motor3.setSpeed(0);
  motor4.setSpeed(0);
  //-------------------------------------|

  // Cut power to motors after they are done running
  motor1.run(RELEASE);
  motor2.run(RELEASE);
  motor3.run(RELEASE);
  motor4.run(RELEASE); 
  delay(10);
  //-------------------------------------|

}