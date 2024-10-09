/**
 * The module accuracy is responsible for checking the accuracy of the model on the arduino
 * To this purpose it sends all the test gestures through the network and compares the resulting label with the desired ones
 * One would need more bytes than available to save all the test gestures
 * Thats why the test set is split into two, with TESTSET you can control which you want to check right now
 */


#ifndef ACCURACY_H
#define ACCURACY_H

//#define TESTSET1
#define TESTSET2


#ifdef TESTSET1
#include "test.h"
#endif
#ifdef TESTSET2
#include "test2.h"
#endif

#include "Arduino.h"
#include "model.h"


/**
 * Take the model and push all the test data through it
 * Prints the results over serial
 */
void checkAccuracy(struct model* myModel);




#endif
