#include "AFMotor_R4.h" // Library to control DC motors on Adafruit shield

// Motor objects for each motor
AF_DCMotor motor1(1); // Right Front motor (M1)
AF_DCMotor motor2(2); // Left Front motor (M2)
AF_DCMotor motor3(3); // Left Back motor (M3)
AF_DCMotor motor4(4); // Right Back motor (M4)

uint8_t turnSpeed = 180;  // Global speed used for turns (0-255)

// Function prototypes
void moveAll(uint8_t direction, uint16_t seconds, uint8_t speed); // Move all motors
void turn(char side, uint16_t seconds); // Turn left or right

void setup() {
  Serial.begin(9600); // Start serial communication with Pi

  motor1.setSpeed(0);
  motor1.run(BRAKE);  // Stop motor1

  motor2.setSpeed(0);
  motor2.run(BRAKE);  // Stop motor2

  motor3.setSpeed(0);
  motor3.run(BRAKE);  // Stop motor3

  motor4.setSpeed(0);
  motor4.run(BRAKE);  // Stop motor4
}

void loop() {
  if (Serial.available() > 0) { // If Pi sent data
    String cmd = Serial.readStringUntil('\n');  // Read full command line
    cmd.trim(); // Remove extra spaces/newlines

    // MOVE command: expects "MOVE F 200 1" or "MOVE B 150 2"
    if (cmd.startsWith("MOVE")) {
      char dir = cmd.charAt(5); // 'F' = forward, 'B' = backward

      int space1 = cmd.indexOf(' ', 6); // First space after direction
      int space2 = cmd.indexOf(' ', space1 + 1);  // Second space after speed

      uint8_t speed = cmd.substring(space1 + 1, space2).toInt();  // Convert speed to number
      uint16_t sec = cmd.substring(space2 + 1).toInt(); // Convert duration to number

      if (dir == 'F') {
        moveAll(FORWARD, sec, speed); // Move forward
      }
      else if (dir == 'B') {
        moveAll(BACKWARD, sec, speed); // Move backward
      }
    }

    // TURN command: expects "TURN L 1" or "TURN R 2"
    else if (cmd.startsWith("TURN")) {
      char side = cmd.charAt(5);  // 'L' = left, 'R' = right
      uint16_t sec = cmd.substring(7).toInt();  // Convert duration to number
      turn(side, sec);  // Execute turn
    }
  }
}

// Move all motors in a direction for a certain speed and time
void moveAll(uint8_t direction, uint16_t seconds, uint8_t speed) {
  motor1.run(direction); // Set right front
  motor2.run(direction); // Set left front
  motor3.run(direction); // Set left back
  motor4.run(direction); // Set right back

  for (uint8_t i = 0; i < speed; i++) { // Gradually ramp up speed
    motor1.setSpeed(i); 
    motor2.setSpeed(i);
    motor3.setSpeed(i); 
    motor4.setSpeed(i);
    delay(10);  // Small delay for smooth ramp
  }

  if (speed == 255) { // Max speed safety
    motor1.setSpeed(255); 
    motor2.setSpeed(255);
    motor3.setSpeed(255); 
    motor4.setSpeed(255);
  }

  delay(seconds * 1000);  // Run motors for specified amount of time

  for (uint8_t i = speed; i > 0; i--) { // Ramp down speed
    motor1.setSpeed(i); 
    motor2.setSpeed(i);
    motor3.setSpeed(i); 
    motor4.setSpeed(i);
    delay(10);
  }

  motor1.setSpeed(0); 
  motor2.setSpeed(0); // Stop motors
  motor3.setSpeed(0); 
  motor4.setSpeed(0);

  motor1.run(RELEASE); 
  motor2.run(RELEASE); // Cut motor power
  motor3.run(RELEASE); 
  motor4.run(RELEASE);
  delay(10); // Short pause
}

// Turn robot left or right
void turn(char side, uint16_t seconds) {
  if (side == 'L') {  // Left turn: right side backward, left side forward
    motor1.run(BACKWARD); 
    motor4.run(BACKWARD); // Right side motors
    motor2.run(FORWARD); 
    motor3.run(FORWARD); // Left side motors
  } 
  else if (side == 'R') { // Right turn: right side forward, left side backward
    motor1.run(FORWARD); 
    motor4.run(FORWARD); // Right side motors
    motor2.run(BACKWARD); 
    motor3.run(BACKWARD); // Left side motors
  } 
  else {  // Invalid input, stop motors
    motor1.run(RELEASE); 
    motor2.run(RELEASE); 
    motor3.run(RELEASE); 
    motor4.run(RELEASE);
    return;
  }

  for (uint8_t i = 0; i < turnSpeed; i++) { // Ramp up turn speed
    motor1.setSpeed(i); 
    motor2.setSpeed(i);
    motor3.setSpeed(i); 
    motor4.setSpeed(i);
    delay(10);
  }

  if (turnSpeed == 255) { // Max speed safety
    motor1.setSpeed(255); 
    motor2.setSpeed(255);
    motor3.setSpeed(255); 
    motor4.setSpeed(255);
  }

  delay(seconds * 1000);  // Turn duration

  for (uint8_t i = turnSpeed; i > 0; i--) { // Ramp down speed
    motor1.setSpeed(i); 
    motor2.setSpeed(i);
    motor3.setSpeed(i); 
    motor4.setSpeed(i);
    delay(10);
  }

  motor1.setSpeed(0); 
  motor2.setSpeed(0); // Stop motors
  motor3.setSpeed(0); 
  motor4.setSpeed(0);

  motor1.run(RELEASE); 
  motor2.run(RELEASE); // Cut motor power
  motor3.run(RELEASE); 
  motor4.run(RELEASE);
  delay(10);  // Short pause after turn
}