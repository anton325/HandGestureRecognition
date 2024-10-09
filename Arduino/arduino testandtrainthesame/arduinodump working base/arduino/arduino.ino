// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "config.h"
#include "predict.h"
#include "model.h"
#include "test3.h"
int16_t Xint[180];


using namespace model;

void accuracy();
void predictionTime();
void syncWithMaster();

void setup() {
	Serial.begin(9600);
 Serial.println("init");
	return;
}

// Sketch operates in two modes: accuracy mode and prediction time mode. Refer to config.h for more details.
void loop() {

#ifdef ACCURACY
	accuracy();
#endif

#ifdef PREDICTIONTIME
	predictionTime();
#endif

	return;
}

// In this mode, first, the sketch synchronizes with the master using a test message.
// Then, each datapoint is read from the serial port and its class ID is predicted.
// The class ID and the prediction time is communicated back to the master using the same serial port.
void accuracy() {

	syncWithMaster();

	while (true) {
		unsigned long startTime = micros();

		int classID = predict();

		unsigned long elapsedTime = micros() - startTime;

		Serial.println(classID);
		Serial.println(elapsedTime);
	}

	return;
}

// In this mode, a single data point is already available on the device's flash storage.
// The data point is stored in the variable X and its class ID is stored in the variable Y.
// Prediction is performed on X and the time per prediction is recorded.
// The average prediction time for 100 runs is computed and is written to the serial port.
  int iterations = 0;
    int wr = 0;
    int* st;

void predictionTime() {
	unsigned long totalTime = 0;
	while(iterations<32) {
    st = (testset+180*(iterations));
    Serial.print("it");
    Serial.println(iterations);
    for(int j = 0;j<180;j++){
      double t = (pgm_read_word(&st[j]))/float(1024);
      double scalet = pow(2,5);
      Xint[j]=(int)(t*pow(2,5));
            //Serial.println(Xint[j]);       
    } 
    
		Serial.print(iterations + 1);
		Serial.print(": ");

		unsigned long startTime = micros();

		int classID = predict();

		unsigned long elapsedTime = micros() - startTime;
		
		Serial.print("Predicted label: ");
		Serial.print(classID);
		Serial.print("; True label: ");
		Serial.print(pgm_read_byte(&testlabel[iterations]));

		if ((classID) == pgm_read_byte(&testlabel[iterations])){
			Serial.println("; Correct prediction");
		}
		else {
			Serial.println("; WARNING: Incorrect prediction");
      wr++;
		}

		totalTime += elapsedTime;
		iterations++;

		if (iterations % 1 == 0) {
			Serial.println("\n------------------------");
			Serial.println("Average prediction time:");
			Serial.println((float)totalTime / iterations);
			Serial.println("------------------------\n");
		}
	}
Serial.println(wr);
	return;
}

// In accuracy mode, this function reads an integer from the serial port.
// In the prediction time mode, this function reads an integer from the array X stored in device's flash memory.
int32_t getIntFeature(MYITE i) {
#ifdef ACCURACY
	char buff[13];
	while (!Serial.available())
		;
	Serial.readBytes(buff, 13);
	double f = (float)(atof(buff));
#endif

#ifdef PREDICTIONTIME
//extern int16_t Xint[180];
  #ifdef XFLOAT
	double f = ((float) pgm_read_float_near(&X[i]));
  #endif
  #ifdef XINT8
  return ((int8_t) pgm_read_byte_near(&Xint[i]));
  #endif
  #ifdef XINT16
  // this is active config
  int16_t temp = Xint[i];
  return ((int16_t) temp);
  #endif
#endif

#ifdef XFLOAT
  double f_int = ldexp(f, -scaleOfX);
  return (int32_t)(f_int);
#endif
}

// Function reads a float variable either from the serial port or from the array X stored in devices' flash memory.
float getFloatFeature(MYINT i) {
#ifdef ACCURACY
	char buff[13];
	while (!Serial.available())
		;
	Serial.readBytes(buff, 13);
	return (float)(atof(buff));
#endif

#ifdef PREDICTIONTIME
	return ((float) pgm_read_float_near(&X[i]));
#endif
}

// Setup serial communication with the computer
void syncWithMaster() {

	// Wait for input
	while (!Serial.available())
		;

	// Match input with the message
	const char *syncMsg = "fixed";
	int len = strlen(syncMsg);
	byte index = 0;
	do {
		char ch = Serial.read();
		if (ch == syncMsg[index])
			index += 1;
	} while (index < len);

	Serial.println("point");

	return;
}
