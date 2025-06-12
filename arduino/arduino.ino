#include <Servo.h>

const int ledPin = 7;
Servo myServo;
int pos = 0;
unsigned long lastFaceTime = 0;
const unsigned long DOOR_DELAY = 3000; // 3 seconds delay
bool doorOpen = false;

void setup() {
  Serial.begin(9600);
  pinMode(ledPin, OUTPUT);
  myServo.attach(9);
  myServo.write(0);  // Assume door starts closed
}

void openDoor() {
  if (!doorOpen) {
    for (pos = 0; pos <= 90; pos++) {
      myServo.write(pos);
      delay(15);
    }
    doorOpen = true;
  }
}

void closeDoor() {
  if (doorOpen) {
    for (pos = 90; pos >= 0; pos--) {
      myServo.write(pos);
      delay(15);
    }
    doorOpen = false;
  }
}

void loop() {
  if (Serial.available() > 0) {
    char data = Serial.read();
    
    if (data == '1') {
      digitalWrite(ledPin, HIGH);
      lastFaceTime = millis(); // Reset timer when face detected
      openDoor();
    }
    else if (data == '0') {
      digitalWrite(ledPin, LOW);
    }
  }
  
  // Check if we should close the door
  if (doorOpen && (millis() - lastFaceTime > DOOR_DELAY)) {
    closeDoor();
  }
} 