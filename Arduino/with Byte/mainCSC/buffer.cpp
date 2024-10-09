/**
   Buffer.cpp
   This cpp file is part of the buffer module
   Please refer to buffer.h for more information
*/


#include "buffer.h"
#include <SoftwareSerial.h>

#define _maxSize 60 // twenty less than total buffer size
#define _alpha 0.01
#define _delta 10.24
#define _margin 0.1
/**
   Constructor of the class buffer
   Initialize the values and constants for this class
   Copy the input values
*/
Buffer::Buffer(void)
{
  Serial.begin(9600); //if debug messages are necessary

  _thresholdLow = 0;
  _thresholdHi = 0;
  _ema = 0;
  _countdownSinceGesture = 0;
  _frameCounter = 0;
  _endBuffer = 0;
  //_maxSize = 60;
}


void Buffer::_resetBuffer() {
  _ema = _meanOfThisFrame;
  // mean of last frame is not useful I think
  _meanOfLastFrame = _meanOfThisFrame;

  _frameCounter = 0;
  _gestureflag = 0;

  // save last frame
  for (int i = 0; i < 9; i++) {
    _lastFrame[i] = _buffer[_endBuffer - 1][i];
  }

  _thresholdLow = _ema * (1 - _margin);
  _thresholdHi = _ema * (1 + _margin);
  _countdownSinceGesture = 0;
  _endBuffer = 0;
}




int Buffer::feedFrame(int* frame) {

  _addFrameToBuffer(frame);
  _meanOfThisFrame =  _getMean(_buffer[_endBuffer - 1]);
  /*
    Serial.println("feedframe");
    Serial.println(_meanOfThisFrame);
    Serial.println(_thresholdLow);
    Serial.println(_thresholdHi);*/

  if (_frameCounter == 0) {
    // this is the first frame, initialize some things
    _ema = _meanOfThisFrame;
    _thresholdLow = _ema * (1 - _margin);
    _thresholdHi = _ema * (1 + _margin);


    for (int i = 0; i < 9 ; i++) {
      _lastFrame[i] = frame[i];
    }
  }

  else if (_meanOfThisFrame >= _thresholdLow && _meanOfThisFrame <= _thresholdHi) {
    // this frame triggers no new event
    if (_gestureflag == 1) {
      // but the frame before was part of an event -> this is the first event after the last that belonged to an event
      _countdownSinceGesture = 3;
      // reset gestureflag
      _gestureflag = 0 ;
    }

    // check if single bits deviate too much from the last frame
    if (_compareFrames(_buffer[_endBuffer - 1], _lastFrame)) {
      // adjust some values
      //Serial.println("Adjust some values");
      _ema = _ema * (1 - _alpha) + _meanOfThisFrame * _alpha;
      _thresholdLow = _ema * (1 - _margin);
      _thresholdHi = _ema * (1 + _margin);
    }
  }

  // check if this frame triggers an event AND no event was recognized before
  else if ((_meanOfThisFrame < _thresholdLow || _meanOfThisFrame > _thresholdHi) && _gestureflag == 0) {
    // This frame triggers new event
    _gestureflag = 1;
    Serial.println("Set gestureflag true");
  }

  // increment the framecounter
  _frameCounter++;

  // decrement countdown
  if (_countdownSinceGesture > 0) {
    _countdownSinceGesture--;
    if (_countdownSinceGesture == 0) {
      // three frames since last frame have passed, tell main that event has happend
      Serial.println("Three frames have passed since last event");
      return 1; // return true
    }
  }


  // when no event is happening, no countdown, keep the buffer at max length of 3
  if (_gestureflag == 0 && _countdownSinceGesture == 0 && _endBuffer > 3) {
    // copy third last to first place, second last to second and last to third place to keep the buffer short, there is no need to save the frames
    for (int i = 0; i < 3; i++) {
      _copyFrame(_buffer[_endBuffer - 3 + i], i);
    }
    // reset the "pointer" to the new end of the buffer
    _endBuffer = 3;
  }
  // check if buffer has gotten too big -> maxSize-3 because we need at least 20 places behind _endBuffer for the pseudoindexes
  if (_endBuffer > (_maxSize)) {
    // buffer too big, hard reset, might lose important information but necessary before it breaks
    Serial.println("reset buffer");
    _resetBuffer();
  }
  return 0; //return false
}


bool Buffer::_compareFrames(int* lastFrame, int* currentFrame) {
  for (int i = 0; i < 9; i++) {
    if (abs(lastFrame[i] - currentFrame[i]) > (_delta)) {
      // it deviates too much!
      return 1;
    }
  }
  // frames are pretty similar, return 0
  return 0;
}


void Buffer::_addFrameToBuffer(int* frame) {
  // copy frame
  for (int i = 0; i < 9; i++) {
    _buffer[_endBuffer][i] = frame[i];
  }
  // increment end buffer
  _endBuffer++;
}



int Buffer::_getMean(int* frame) {
  double mean = 0;
  for (int i = 0; i < 9; i++) {
    mean += frame[i];
  }
  mean /= 9;
  return mean;
}


void Buffer::printBuffer() {
  Serial.println("Gebe Buffer aus: ");
  for (int i = 0; i < _endBuffer; i++) {
    Serial.println("Frame:");
    for (int j = 0; j < 9; j++) {
      Serial.print(_buffer[i][j]);
      Serial.print(" ");
    }
  }
}


void Buffer::printWholeBuffer() {
  Serial.println("Gebe Buffer aus: ");

  for (int i = 0; i < 30; i++) {
    Serial.println(" ");
    Serial.print("Frame ");
    Serial.print(i);
    Serial.print(": ");
    for (int j = 0; j < 9; j++) {
      Serial.print(_buffer[i][j]);
      Serial.print(" ");
    }
  }
  Serial.println(" ");
  double* h = (double*) _buffer[30];
  for (int i = 0; i < 7; i++) {
    Serial.println(" ");
    Serial.print("Frame ");
    Serial.print(i);
    Serial.print(": ");
    for (int j = 0; j < 9; j++) {
      Serial.print(h[i * 9 + j]);
      Serial.print(" ");
    }
  }
  Serial.println(" ");
  h = (double*) _buffer[37];
  for (int i = 0; i < 20; i++) {
    Serial.println(" ");
    Serial.print("Frame ");
    Serial.print(i);
    Serial.print(": ");
    for (int j = 0; j < 9; j++) {
      Serial.print(h[i * 9 + j]);
      Serial.print(" ");
    }
  }
  Serial.println(" ");
  h = (double*) _buffer[0];
  for (int i = 0; i < 20; i++) {
    Serial.println(" ");
    Serial.print("Frame ");
    Serial.print(i);
    Serial.print(": ");
    for (int j = 0; j < 9; j++) {
      Serial.print(h[i * 9 + j]);
      Serial.print(" ");
    }
  }
  Serial.println(" ");
}


void Buffer::_copyFrame(int* frame, int index) {
  for (int i = 0; i < 9; i++) {
    _buffer[index][i] = frame[i];
  }
}





double* Buffer::getScaledBuffer() {
  //Serial.print("Buffer index at:");
  //Serial.println(_endBuffer);
  // let the scaled buffer start behind at the very end
  double* startOfScaledBuffer = (double*) _buffer[37];
  //scale all frames and write them to the assigned spot
  for (int i = 0; i < 20; i++) {
    pseudoindex = i * (_endBuffer - 1) / float(19);
    int low = floor(pseudoindex);
    int high = ceil(pseudoindex);
    // this interpolated frame goes behind the pseudoindexes
    for (int j = 0; j < 9; j++) {
      if (low == high) {
        startOfScaledBuffer[i * 9 + j] = (_buffer[low][j])/float(1024);
      }
      else {
        double d1 = pseudoindex - int(pseudoindex);
        double d2 = 1 - d1;
        startOfScaledBuffer[i * 9 + j] = (_buffer[low][j] * d2 + _buffer[high][j] * d1)/float(1024);
      }
    }
  }
  _resetBuffer();
  return startOfScaledBuffer;
}

int* Buffer::rawScene() {
  return _buffer[0];
}
