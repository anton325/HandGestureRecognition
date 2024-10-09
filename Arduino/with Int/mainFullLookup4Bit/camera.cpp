/**
 * camera.cpp
 * The implementation to the camera.h file of the camera module
 * For more information refer to "camera.h"
 */


#include "camera.h"

void readRow(int row,double* frame){
   // Activate pin that supplys voltage to the row.
  digitalWrite(row + OUT_PIN_MIN, HIGH);

  // Wait some time. 
  delay(DELAY_PER_ROW);

  // Read add values of the row from the analog input
  for (int col = 0; col < NO_COLS; col++) {
    int brightnessValue = analogRead(col + AIN_PIN_MIN);
    // Serial.println(brightnessValue);
    frame[NO_COLS*row+col] = brightnessValue; 
  }

  // Deactivate pin that supplys voltage to the row.
  digitalWrite(row + OUT_PIN_MIN, LOW);
  
}


void readFrame(double* frame){
  Serial.begin(9600);
    // Read all rows 
  for (int row = 0; row < NO_ROWS; row++) {
    readRow(row,frame); 
  }
}
