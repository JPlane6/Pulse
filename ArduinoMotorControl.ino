#include "AFMotor_R4.h" // Import Adafruit Motor Driver Library

// Motor mapping: right side = M1 & M4, left side = M2 & M3
AF_DCMotor motor1(1);  // Right front
AF_DCMotor motor2(2);  // Left front
AF_DCMotor motor3(3);  // Left back
AF_DCMotor motor4(4);  // Right back

// Global Speed Variables
uint8_t turnSpeed = 180; // Speed for turning (0-255)

// Prototype Methods
void moveAll(uint8_t direction ,uint16_t seconds, uint8_t speed);
void turn(char dir ,uint16_t seconds); // 'L' for left, 'R' for right

void setup() {

  // Initialize motors: start stopped and in brake mode
  motor1.setSpeed(0); 
  motor1.run(BRAKE);

  motor2.setSpeed(0); 
  motor2.run(BRAKE);

  motor3.setSpeed(0); 
  motor3.run(BRAKE);

  motor4.setSpeed(0); 
  motor4.run(BRAKE);

}

void loop() {
  
  // Test sequence: move forward/backward and turn left/right
  moveAll(FORWARD, 1, 200);
  moveAll(BACKWARD, 1, 200);
  turn('L', 1);
  turn('R', 1);

  while(true){} // Stop further action after test sequence

}

// Smoothly move all four motors forward/backward
void moveAll(uint8_t direction ,uint16_t seconds, uint8_t speed) {

  // Set all motors to chosen direction
  motor1.run(direction);
  motor2.run(direction);
  motor3.run(direction);
  motor4.run(direction);

  // Gradually ramp up motor speed for smoother start
  for (uint8_t i = 0; i < speed; i++) {
    motor1.setSpeed(i);
    motor2.setSpeed(i);
    motor3.setSpeed(i);
    motor4.setSpeed(i);
    delay(10); // Small delay to smooth acceleration
  }

  // Safety: ensure max speed is properly set
  if (speed == 255) {
    motor1.setSpeed(255); 
    motor2.setSpeed(255);
    motor3.setSpeed(255);
    motor4.setSpeed(255);
  }

  delay(seconds * 1000);  // Maintain motion for specified duration

  // Gradually ramp down motor speed for smoother stop
  for (uint8_t i = speed; i > 0; i--) {
    motor1.setSpeed(i);
    motor2.setSpeed(i);
    motor3.setSpeed(i);
    motor4.setSpeed(i);
    delay(10); // Small delay to smooth deceleration
  }

  // Ensure motors are fully stopped and power released
  motor1.setSpeed(0); 
  motor2.setSpeed(0);
  motor3.setSpeed(0); 
  motor4.setSpeed(0);

  motor1.run(RELEASE); 
  motor2.run(RELEASE);
  motor3.run(RELEASE); 
  motor4.run(RELEASE);
  delay(10);

}

// Smoothly rotate robot left or right using turnSpeed
void turn(char dir ,uint16_t seconds) {

  // Set individual motor directions based on turn side
  if (dir == 'L') {  // Left turn: left motors backward, right motors forward
    motor2.run(BACKWARD); // Left front
    motor3.run(BACKWARD); // Left back
    motor1.run(FORWARD);  // Right front
    motor4.run(FORWARD);  // Right back
  } 
  else if (dir == 'R') { // Right turn: left motors forward, right motors backward
    motor2.run(FORWARD);   // Left front
    motor3.run(FORWARD);   // Left back
    motor1.run(BACKWARD);  // Right front
    motor4.run(BACKWARD);  // Right back
  } 
  else {  // Invalid input: safely release all motors
    motor1.run(RELEASE);
    motor2.run(RELEASE);
    motor3.run(RELEASE);
    motor4.run(RELEASE);
    return;
  }

  // Ramp up motor speed for smooth turning
  for (uint8_t i = 0; i < turnSpeed; i++) {
    motor1.setSpeed(i);
    motor2.setSpeed(i);
    motor3.setSpeed(i);
    motor4.setSpeed(i);
    delay(10);
  }

  // Safety: ensure max speed is properly applied
  if (turnSpeed == 255) {
    motor1.setSpeed(255);
    motor2.setSpeed(255);
    motor3.setSpeed(255);
    motor4.setSpeed(255);
  }

  delay(seconds * 1000); // Maintain turn for specified duration

  // Ramp down motors to stop turn smoothly
  for (uint8_t i = turnSpeed; i > 0; i--) {
    motor1.setSpeed(i);
    motor2.setSpeed(i);
    motor3.setSpeed(i);
    motor4.setSpeed(i);
    delay(10);
  }

  // Stop motors and cut power after turn
  motor1.setSpeed(0); 
  motor2.setSpeed(0);
  motor3.setSpeed(0); 
  motor4.setSpeed(0);

  motor1.run(RELEASE); 
  motor2.run(RELEASE);
  motor3.run(RELEASE); 
  motor4.run(RELEASE);
  delay(10);

}