//只能在Arduino DUE上跑

#include <Wire.h>
int input;
void setup() {
  Serial.begin(115200);
  pinMode(A3, INPUT);
  pinMode(DAC0,OUTPUT);
  Wire.begin();
  Wire.setClock(440000);//11000
  analogReadResolution(12);
  analogWriteResolution(12);
}

void loop() {
  analogWrite(DAC0,analogRead(A3));
}
