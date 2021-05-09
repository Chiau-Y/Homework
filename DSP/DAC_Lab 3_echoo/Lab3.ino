//只能在Arduino DUE上執行程式

#include <Wire.h>
int input, output;
int i = 0, j = 1, fs = 2250;
int d = 18000;//d=fs*dt
int x1, x1s[18000];
void setup() {
  pinMode(A3, INPUT);
  pinMode(DAC0, OUTPUT);
  Wire.begin();
  Wire.setClock(fs);
  analogReadResolution(12);
  analogWriteResolution(12);
  Serial.begin(9600);
}
void loop() {
  x1 = analogRead(A3);
  analogWrite(DAC0, x1); //no echo
  x1s[i++] = x1;
  if (i == d) {
    i = 0;
    while (true) {
      x1 = analogRead(A3);
      x1s[i++] = x1 + 0.6 * x1s[j];
      analogWrite(DAC0, x1s[i]); //one echo
      Serial.println(x1s[i]);
      if (j == d) {
        j = 0;
      } else {
        j = i + 1;
      }
      if (i >= d) {
        i = 0;
      }
    }
  }
}
