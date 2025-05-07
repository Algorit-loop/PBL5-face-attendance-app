#include <Servo.h>

const int ledPin = 7;
Servo myServo;
int pos = 0;

void setup() {
  Serial.begin(9600);
  pinMode(ledPin, OUTPUT);
  myServo.attach(9);
  myServo.write(0);  // Assume door starts closed
}

void loop() {
  if (Serial.available() > 0) {
    char data = Serial.read();
    
    if (data == '1') {
      digitalWrite(ledPin, HIGH);
    }
    else if (data == '0') {
      digitalWrite(ledPin, LOW);
    }
    else if (data == 'o') {
      // Open door: 
      for (pos = 90; pos >= 0; pos--) {
        myServo.write(pos);
        delay(15);
      }
    }
    else if (data == 'c') {
      for (pos = 0; pos <= 90; pos++) {
        myServo.write(pos);
        delay(15);
      }
    }
  }
}
