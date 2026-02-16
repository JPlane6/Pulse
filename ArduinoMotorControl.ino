int motor1Forward = 2; // connected internally to shield
int motor1Backward = 3;

void setup() {
  Serial.begin(9600);
  pinMode(motor1Forward, OUTPUT);
  pinMode(motor1Backward, OUTPUT);
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();
    switch(cmd) {
      case 'f': // forward
        digitalWrite(motor1Forward, HIGH);
        digitalWrite(motor1Backward, LOW);
        break;
      case 'b': // backward
        digitalWrite(motor1Forward, LOW);
        digitalWrite(motor1Backward, HIGH);
        break;
      case 's': // stop
        digitalWrite(motor1Forward, LOW);
        digitalWrite(motor1Backward, LOW);
        break;
    }
  }
}
