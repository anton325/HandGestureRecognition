/**
 * Buffer.cpp
 * This cpp file is part of the buffer module
 * Please refer to buffer.h for more information
 */


#include "buffer.h"
#include <SoftwareSerial.h>
/**
   Constructor of the class buffer
   Initialize the values and constants for this class
   Copy the input values
*/
Buffer::Buffer(double alpha, double delta, double margin)
{
  Serial.begin(9600); //if debug messages are necessary

  _alpha = alpha;
  _delta = delta;
  _margin = margin;
  _thresholdLow = 0;
  _thresholdHi = 0;
  _ema = 0;
  _countdownSinceGesture = 0;
  _frameCounter = 0;
  _endBuffer = 0;
  _maxSize = 39;
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




int Buffer::feedFrame(double* frame) {
  _addFrameToBuffer(frame);
  _meanOfThisFrame =  _getMean(_buffer[_endBuffer - 1]);


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
      //Serial.println("Three frames have passed since last event");
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
  if (_endBuffer > (_maxSize-3)) {
    // buffer too big, hard reset, might lose important information but necessary before it breaks    
    Serial.println("reset buffer");
    _resetBuffer();
  }
  return 0; //return false
}


bool Buffer::_compareFrames(double* lastFrame, double* currentFrame) {
  for (int i = 0; i < 9; i++) {
    if (abs(lastFrame[i] - currentFrame[i]) > _delta) {
      // it deviates too much!
      return 1;
    }
  }
  // frames are pretty similar, return 0
  return 0;
}


void Buffer::_addFrameToBuffer(double* frame) {
  // copy frame
  for (int i = 0; i < 9; i++) {
    _buffer[_endBuffer][i] = frame[i] / 1024;
  }
  // increment end buffer
  _endBuffer++;
}



double Buffer::_getMean(double* frame) {
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
  
  for (int i = 0; i < 36; i++) {
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
  for(int i = 0; i<36; i++){
    for(int j = 0; j<9;j++){
      Serial.print(_buffer[i][j]);
      Serial.print(",");
    }
  }
}


void Buffer::_copyFrame(double* frame, int index) {
  for (int i = 0; i < 9; i++) {
    _buffer[index][i] = frame[i];
  }
}



double* Buffer::getScaledBuffer() {
  Serial.print("Buffer index at:");
  Serial.println(_endBuffer);

  // let pseudoindexes point to end of buffer, it should at least be behind the first 20 frames because thats where we want to store the scaled buffer in the end
  double* startOfPseudoIndexes;

  if(_endBuffer<20){
    startOfPseudoIndexes = _buffer[21];
  }
  else{
    startOfPseudoIndexes = _buffer[_endBuffer]; // let it point behind the buffer, because it is longer than 20
  }
  
  // calcualte pseudoindexes
  for (int i = 0; i < 20; i++) {
    startOfPseudoIndexes[i] = i * (_endBuffer - 1) / float(19);
  }

  // calculate the interpolated frames AND put a certain number of them behind the pseudoindexes
  // this is important again, because I want to reuse the _buffer again, so to do that we need to put the first interpolated frames BEHIND the pseudoindexes
  // if we put them at the front we overwrite important information
  // After putting a certain number of frames behind the pseudoindexes we can put the remaining at the front without fear of overwriting frames

  double* startOfScaledBuffer = (startOfPseudoIndexes + 20);
  // determine how many frames go behind the pseudoindexes
  int framesThatGoBehindBuffer = 0;
  if(_endBuffer<20){  // if buffer is not that long, we can put a lot of stuff behind the buffer and we also need to
    framesThatGoBehindBuffer = 20-_endBuffer;
  }
  else{ // end buffer is pretty long, so we cant put too many frames behind the buffer
    framesThatGoBehindBuffer = 5;
  }
  

  for (int i = 0; i < 20; i++) {
    int low = floor(startOfPseudoIndexes[i]);
    int high = ceil(startOfPseudoIndexes[i]);
    
    if (i < framesThatGoBehindBuffer) {
      // this interpolated frame goes behind the pseudoindexes
      for (int j = 0; j < 9; j++) {
        if (low == high) {
          startOfScaledBuffer[i * 9 + j] = _buffer[int(startOfPseudoIndexes[i])][j];
        }
        else {
          double d1 = startOfPseudoIndexes[i] - int(startOfPseudoIndexes[i]);
          double d2 = 1 - d1;
          startOfScaledBuffer[i * 9 + j] = _buffer[(int)floor(startOfPseudoIndexes[i])][j] * d2 + _buffer[(int)ceil(startOfPseudoIndexes[i])][j] * d1;
        }
      }
    }
    else {
     // this interpolated frame goes at the beginning of the buffer
      for (int j = 0; j < 9; j++) {
        if (low == high) {
          _buffer[i - framesThatGoBehindBuffer][j] = _buffer[int(startOfPseudoIndexes[i])][j];
        }
        else {
          double d1 = startOfPseudoIndexes[i] - int(startOfPseudoIndexes[i]);
          double d2 = 1 - d1;
          _buffer[i - framesThatGoBehindBuffer][j] = _buffer[(int)floor(startOfPseudoIndexes[i])][j] * d2 + _buffer[(int)ceil(startOfPseudoIndexes[i])][j] * d1;
        }
      }
    }
  }

  // now we need to put them back in the correct order
  // first push the ones at the beginning back
  for (int i = 20 - framesThatGoBehindBuffer; i >= 0; i--) {
    for (int j = 0; j < 9 ; j++) {
      _buffer[framesThatGoBehindBuffer + i][j] = _buffer[i][j];
    }
  }

  // now put the ones behind the pseudoindexes at the front
  for (int i = 0; i < framesThatGoBehindBuffer; i++) {
    for (int j = 0; j < 9; j++) {
      _buffer[i][j] = startOfScaledBuffer[i * 9 + j];
    }
  }

  // now they are in order and we can return the beginning because thats where they are
  double* startOfBuffer = &_buffer[0][0];
  return startOfBuffer;
}


double* Buffer::rawScene(){
  return _buffer[0];
}
