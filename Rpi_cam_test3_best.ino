#include <Servo.h>

Servo servoPan;
Servo servoTilt;

int panPin = 10;
int tiltPin = 11;

int currentPanAngle = 60;
int currentTiltAngle = 60;
int targetPanAngle;
int targetTiltAngle;

const int maxChange = 1;
const int delayTime = .001;
const int threshold = 5;

String inputString = "";
boolean stringComplete = false;

void setup() {
  Serial.begin(115200);
  servoPan.attach(panPin);
  servoTilt.attach(tiltPin);

  // put your setup code here, to run once:
  servoPan.write(currentPanAngle);
  servoTilt.write(currentTiltAngle);
}

void loop() {
  if (stringComplete) {
    parseInputString();
    moveServos(targetPanAngle, targetTiltAngle);

    inputString = "";
    stringComplete = false;
  }
  // put your main code here, to run repeatedly:

}

void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n'){
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }
}

void parseInputString() {
  int commaIndex = inputString.indexOf(',');
  if (commaIndex > 0) {
    int centerX = inputString.substring(0, commaIndex).toInt();
    int centerY = inputString.substring(commaIndex + 1).toInt();
    targetPanAngle = map(centerX, 0, 1280, 180, 0);
    targetTiltAngle = map(centerY, 0, 720, 0, 180);
  }
}

void moveServos(int targetPan, int targetTilt) {
  if (abs(currentPanAngle - targetPan) > threshold){
  if (currentPanAngle < targetPan) {
    currentPanAngle = min(currentPanAngle + maxChange, targetPan);
  } else if (currentPanAngle > targetPan) {
    currentPanAngle = max(currentPanAngle - maxChange, targetPan);
  }
  
  servoPan.write(currentPanAngle);
  }
  if (abs(currentTiltAngle - targetTilt) > threshold) {
  if (currentTiltAngle < targetTilt) {
    currentTiltAngle = min(currentTiltAngle + maxChange, targetTilt);
  } else if (currentTiltAngle > targetTilt) {
    currentTiltAngle = max(currentTiltAngle - maxChange, targetTilt);
  }
  servoTilt.write(currentTiltAngle);
  }

  delay(delayTime);
  }