#include <Wire.h>
int fs = 44000;
double x0, x1 = 0, x2 = 0, x3 = 0, x4 = 0, _y0 = 0, _y1 = 0, y2 = 0, y3 = 0, y4 = 0;
void setup() {
  pinMode(A3 , INPUT);                       //ADC
  pinMode(DAC1, OUTPUT);                     //DAC
  Wire.begin();
  Wire.setClock(fs);
  analogReadResolution(12);
}

void loop() {
  x0 = analogRead(A3);                       //ADC
  //---------------Filter---------------//
  _y0 = 2.109 * y3 - 3.923 * y2 + 3.237 * _y1 - 0.4245 * y4 + 0.06172 * x2 - 0.1234 * x3 + 0.06172 * x4 ;
  analogWrite(DAC1, _y0);                    //DAC
  x4 = x3;
  x3 = x2;
  x2 = x1;
  x1 = x0;
  y4 = y3;
  y3 = y2;
  y2 = _y1;
  _y1 = _y0;
}
