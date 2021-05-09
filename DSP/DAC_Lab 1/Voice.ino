#include <Wire.h>
#include <Adafruit_MCP4725.h>
Adafruit_MCP4725 dac;
int input;
void setup() {
  pinMode(A0, INPUT);
  Wire.begin();
  Wire.setClock(480000);
  analogReadResolution(12);
  dac.begin(0x60);
}

void loop() {
  input = analogRead(A0);
  dac.setVoltage(input, false);
}
