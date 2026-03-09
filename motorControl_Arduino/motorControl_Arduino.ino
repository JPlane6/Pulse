#include "AFMotor_R4.h" // Library to control DC motors on Adafruit shield

// Motor objects for each motor
AF_DCMotor motor1(4); // Back Right (M4)
AF_DCMotor motor2(3); // Back Left (M3)
AF_DCMotor motor3(2); // Front Right (M2)
AF_DCMotor motor4(1); // Front Left (M1)

uint8_t turnSpeed = 180;  // Global speed used for turns (0-255)

// State machine states
enum State { IDLE, RAMPING_UP, RUNNING, RAMPING_DOWN };
State state = IDLE;

// Tracking variables for non-blocking timing
uint8_t targetSpeed = 0;
uint8_t currentRampSpeed = 0;
unsigned long stateStartTime = 0;
unsigned long runDuration = 0;
unsigned long lastRampTime = 0;
bool locked = false; // Locked after STOP, unlocked by GO

// Function prototypes
void moveAll(uint8_t direction, uint16_t seconds, uint8_t speed); // Move all motors
void turn(char side, uint16_t seconds); // Turn left or right

void setAllSpeed(uint8_t spd) {
  motor1.setSpeed(spd);
  motor2.setSpeed(spd);
  motor3.setSpeed(spd);
  motor4.setSpeed(spd);
}

void stopAll() {
  motor1.run(BRAKE); // Stop motor1
  motor2.run(BRAKE); // Stop motor2
  motor3.run(BRAKE); // Stop motor3
  motor4.run(BRAKE); // Stop motor4
  state = IDLE;
  Serial.println("STOPPED");
}

void releaseAll() {
  motor1.run(RELEASE); // Cut motor power
  motor2.run(RELEASE); // Cut motor power
  motor3.run(RELEASE); // Cut motor power
  motor4.run(RELEASE); // Cut motor power
  state = IDLE;
  Serial.println("READY");
}

void setup() {
  Serial.begin(9600); // Start serial communication with Pi

  motor1.run(BRAKE);  // Stop motor1
  motor2.run(BRAKE);  // Stop motor2
  motor3.run(BRAKE);  // Stop motor3
  motor4.run(BRAKE);  // Stop motor4

  Serial.println("READY");
}

void loop() {
  if (Serial.available() > 0) { // If Pi sent data
    String cmd = Serial.readStringUntil('\n');  // Read full command line
    cmd.trim(); // Remove extra spaces/newlines

    // STOP works always, even mid-move
    if (cmd == "STOP") {
      stopAll();
      locked = true; // Lock robot until GO is received
      Serial.println("LOCKED");
      return;
    }

    // GO unlocks the robot after a STOP
    if (cmd == "GO") {
      locked = false;
      Serial.println("UNLOCKED");
      return;
    }

    // MOVE command: expects "MOVE F 200 1" or "MOVE B 150 2"
    if (state == IDLE && !locked) {
      if (cmd.startsWith("MOVE")) {
        char dir = cmd.charAt(5); // 'F' = forward, 'B' = backward

        int space1 = cmd.indexOf(' ', 6); // First space after direction
        int space2 = cmd.indexOf(' ', space1 + 1);  // Second space after speed

        uint8_t speed = cmd.substring(space1 + 1, space2).toInt();  // Convert speed to number
        uint16_t sec = cmd.substring(space2 + 1).toInt(); // Convert duration to number

        if (dir == 'F') {
          moveAll(FORWARD, sec, speed); // Move forward
        } else if (dir == 'B') {
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

  // Non-blocking state machine - replaces all delay() calls
  unsigned long now = millis();

  if (state == RAMPING_UP) {
    if (now - lastRampTime >= 10) { // Every 10ms, same as your original delay(10)
      lastRampTime = now;
      currentRampSpeed++;
      setAllSpeed(currentRampSpeed);

      if (currentRampSpeed >= targetSpeed) { // Max speed safety
        setAllSpeed(targetSpeed);
        stateStartTime = millis();
        state = RUNNING;
      }
    }
  }

  else if (state == RUNNING) {
    if (now - stateStartTime >= runDuration) { // Same as your delay(seconds * 1000)
      currentRampSpeed = targetSpeed;
      lastRampTime = now;
      state = RAMPING_DOWN;
    }
  }

  else if (state == RAMPING_DOWN) {
    if (now - lastRampTime >= 10) { // Every 10ms, same as your original delay(10)
      lastRampTime = now;
      if (currentRampSpeed > 0) {
        currentRampSpeed--;
        setAllSpeed(currentRampSpeed);
      } else {
        releaseAll(); // Cut motor power
        Serial.println("DONE");
      }
    }
  }
}

// Move all motors in a direction for a certain speed and time
void moveAll(uint8_t direction, uint16_t seconds, uint8_t speed) {
  motor1.run(direction); // Set right front
  motor2.run(direction); // Set left front
  motor3.run(direction); // Set left back
  motor4.run(direction); // Set right back

  targetSpeed = speed;
  runDuration = (unsigned long)seconds * 1000;
  currentRampSpeed = 0;
  setAllSpeed(0);
  lastRampTime = millis();
  state = RAMPING_UP;
  Serial.println("MOVING");
}

// Turn robot left or right
void turn(char side, uint16_t seconds) {
  if (side == 'L') {  // Left turn: right side backward, left side forward
    motor1.run(BACKWARD);
    motor4.run(BACKWARD); // Right side motors
    motor2.run(FORWARD);
    motor3.run(FORWARD); // Left side motors
  } else if (side == 'R') { // Right turn: right side forward, left side backward
    motor1.run(FORWARD);
    motor4.run(FORWARD); // Right side motors
    motor2.run(BACKWARD);
    motor3.run(BACKWARD); // Left side motors
  } else {  // Invalid input, stop motors
    motor1.run(RELEASE);
    motor2.run(RELEASE);
    motor3.run(RELEASE);
    motor4.run(RELEASE);
    return;
  }

  targetSpeed = turnSpeed;
  runDuration = (unsigned long)seconds * 1000;
  currentRampSpeed = 0;
  setAllSpeed(0);
  lastRampTime = millis();
  state = RAMPING_UP;
  Serial.println("TURNING");
}