/**
 * camera.h
 * Skeleton for the camera module
 * Utilizes the camera to read frames and pass them to the main module
 * We need to define some definitions at the beginning ie where the pins
 * are and how many rows/columns there are
 */

#ifndef CAMERA_H
#define CAMERA_H
#include "arduino.h"

#define OUT_PIN_MIN 2
#define OUT_PIN_MAX 4

#define AIN_PIN_MIN A0
#define AIN_PIN_MAX A2

#define IN_PIN_MIN 8
#define IN_PIN_MAX 13

// Time in ms to wait per row, that is, between the output pin is set an the analog inputs are read.
#define DELAY_PER_ROW 2 //3

// Number of rows of a frame
#define NO_ROWS (OUT_PIN_MAX - OUT_PIN_MIN + 1)

// Number of rows of a frame
#define NO_COLS (AIN_PIN_MAX - AIN_PIN_MIN + 1)

// Number of pixels of a frame
#define NO_PIXELS (NO_ROWS * NO_COLS)


/**
 * readRow
 * Reads a single row
 * @param int row: defines which row we want to read
 * @param double* frame: points at the beginning of the frame array
 */
void readRow(int row,double* frame);

/**
 * readFrame
 * reads the whole frame by calling readrow repeatedly
 * @param double* frame: points at the beginning of the frame array
 * @returns the frame array
 */
void readFrame(double* frame);
#endif // CAMERA_H
