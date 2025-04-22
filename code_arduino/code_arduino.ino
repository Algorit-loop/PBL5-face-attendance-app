const int ledPin = 7; // Built-in LED (or change to your pin)

void setup() {
  Serial.begin(9600);
  pinMode(ledPin, OUTPUT);
}

void loop() {
  if (Serial.available() > 0) {
    char data = Serial.read();
    if (data == '1') {
      digitalWrite(ledPin, HIGH); // Turn LED ON  // Then OFF
    }
  }
}
