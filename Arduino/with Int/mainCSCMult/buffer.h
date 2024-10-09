/**
 * Buffer.h
 * The buffer module (consisting of buffer.cpp and buffer.h) contains the buffer class which is a replica of the buffer module 
 * created by kubik in python.
 * New frames get append and when the integrated gesture detection unit gets triggered, a signal to the main class is given
 * Then the frames of the gesture get scaled and delivered to the main class for further processing
 */



#ifndef BUFFER_H
#define BUFFER_H


#include "Arduino.h"
#include <stdint.h>

// declare buffer class
class Buffer
{
  // public variables and functions
  public:
    //constructor
    Buffer(double alpha, double delta, double margin);

    /**
     * feedframe
     * FeedFrame accepts the input frame and saves it in the _buffer array
     * Furthermore it detects the start of an event
     * @param double* frame: the input frame to be saved
     * @return int: Return 1 when an event has finished
     */
    int feedFrame(double* frame);

    /**
     * printBuffer
     * exists for debug purpose, it is capable of printing the buffer starting from zero up to the "end pointer" which acts as limit
     */
    void printBuffer();
    
    /**
     * printwholeBuffer
     * Similar to printbuffer, the difference is that printwholeBuffer outputs the whole array without stopping at the limit. This is useful
     * for looking at the calcualted pseudoindexes since the are located just behind the end of the "official" part of the buffer
     */
    void printWholeBuffer();

    /**
     * getScaledBuffer
     * Gets called by the main when feedframe returns true
     * At this time only the frames of the actual gesture are in the buffer. There might be more or less than the required 20 frames. 
     * This function scales them down to 20 using pseudoindexes
     * @return double*: pointer to the start of the 20 frames
     */
    double* getScaledBuffer();
    


    
  // private variables and functions
  private:
  /**
   * _addFrameToBuffer
   * @param double* frame: the newly recorded frame
   * Gets normalized and added to the buffer
   * The _endBuffer gets incremented
   */
    void _addFrameToBuffer(double* frame);

    /**
     * _getMean
     * Capable of calculating the mean of a frame
     * @return double: mean value
     */
    double _getMean(double* frame);

    /**
     * _resetBuffer()
     * When a gesture lasts longer than the memory space reserved, the buffer gets reset
     */
    void _resetBuffer();

    /**
     * _compareFrames
     * A function necessary for the gesture detection
     * @param double* lastFrame: the lastFrame
     * @param double* currentFrame: the current Frame
     * @return bool: Returns true or false, depends on how similiar the frames are
     */
    bool _compareFrames(double* lastFrame, double* currentFrame);

    /**
     * _copyFrame
     * Helper function to copy one frame to a certain index of the buffer
     * Necessary for _addFrameToBuffer, because while no gesture gets recorded the first frame gets deleted and the frames left copied one space to the front
     */
    void _copyFrame(double* frame, int index);
    

    /**
     * private variables, important for the gesture detection
     */
    double _tresholdLow;
    double _tresholdHi;
    double _ema;
    double _alpha;
    double _delta;
    double _margin;

    double _meanOfThisFrame;
    double _meanOfLastFrame;

    bool _gestureflag;

    int _countdownSinceGesture;
    int _frameCounter;

    /**
     * The buffer has a capacity of 34 (_maxSize) frames
     * This is due to memory space restrictions
     */
    double _buffer[32][9]; // max 39
    //double pseudoindexes[20];

    /**
     * _endBuffer keeps track to which index the buffer is currently filled with frames
     */
    int _endBuffer;
    int _maxSize;

    /**
     * the pointer pseudoindexes gets used later on to point behind the last frame in _buffer to save the pseudoindexes there
     */
    double* _pseudoindexes;

    double _lastFrame[9];

    
    
    
};




#endif
