/**
  @file    model.c
  @author  Anton
  @version 0.1
  @date    11.9.2020

  @brief   The implementation of a feedforward neural network
*/

// Do not write above this line (except comments)!
/* SECTION 1: Included header files to compile this file           */
#include "model.h"
#include "model_properties_CSCMultByte.h"
#include <SoftwareSerial.h>


/**
   public function
   Calls the activation of outputlayer function to get the activation
   Then it checks at which neuron we have the highest value -> the index is the output
*/
int evaluateInput(struct model* myModel, double* input) {
  // get output
  //double* out = activationOfOutputlayer(myModel, input
  // reuse input pointer to save memory
   input = activationOfOutputlayer(myModel, input);

  //find index of max neuron

  //double maxActivation = -100;
  //save maxActivation in input[myModel->numOutputs] to save memory
  input[myModel->numOutputs] = -100;
  int indexOfMax = -1;
  for (int index = 0; index < myModel->numOutputs; index++) { // iterate through all outputs
  //Serial.println(input[index]);

    if (input[index] > input[myModel->numOutputs]) {
      input[myModel->numOutputs] = input[index];
      indexOfMax = index;
    }
  }
  // return only the one index
  return indexOfMax;
}



/**
   activation of outputlayer
   Calculates the activation of the last layer!
   It uses the input, iterates through the layers
   Sum Products of input and weights, add bias
   At the end of each layer, the input is set to the activation of the layer we are about to leave
*/
double* activations;
double weight;
unsigned char row;
int currentColumn;
double sumInputs;
int labelCount;
int labelsAddedInARow;
unsigned char labelIndex;

double* activationOfOutputlayer(struct model* myModel, double* input) {
  // save number of Inputs and number of layers
  int numInputs = myModel->numInputs;
  
  //make activations point behind incoming puffer, to save memory
  activations = (input+180);
  

  //iterate over the layers
  for (int layerCounter = 0; layerCounter < myModel->layers; layerCounter++) {


    // set all the activations zero
    for (int j = 0; j < myModel->layers[layerCounter].numNeurons; j++) {
      activations[j] = 0;
    }

    // iterate over the data in this layer
    // keep track of the current column
    currentColumn = 0;
    labelCount = 0;
    labelsAddedInARow = 0;
    sumInputs = 0;
    unsigned char labelptrcontent = pgm_read_byte(&myModel->layers[layerCounter].labelptr[labelCount]);
    for(int index = 0 ; index < myModel->layers[layerCounter].numberOfNonZeroValues; index++){
      if(labelsAddedInARow >= labelptrcontent){
        // apply multiplication
        labelIndex = pgm_read_byte(&myModel->layers[layerCounter].labels[labelCount]);
        weight = pgm_read_float(&myModel->layers[layerCounter].centroids[labelIndex]);
        activations[currentColumn] += weight* sumInputs;
        labelCount++;
        labelptrcontent = pgm_read_byte(&myModel->layers[layerCounter].labelptr[labelCount]); // update labelptrcontent, since labelCount was iterated
        sumInputs = 0;
        labelsAddedInARow = 0;
      }
      while (index >= pgm_read_word(&myModel->layers[layerCounter].indptr[currentColumn + 1])){
        // if yes, increment
        currentColumn ++ ;
      }
      
      // check in which row the original maxtrix the non zero value would have been
      row = (pgm_read_byte(&myModel->layers[layerCounter].indices[index]));
      sumInputs += input[row];
      labelsAddedInARow++;     
    }
    
    labelIndex = (pgm_read_byte(&myModel->layers[layerCounter].labels[labelCount]));
    activations[currentColumn] += sumInputs * pgm_read_float(&myModel->layers[layerCounter].centroids[labelIndex]);
    /*
    for (int i = 0; i < myModel->layers[layerCounter].numNeurons; i++) {
      //input[i] = activations[i];
    //  Serial.println(input[i]);
      Serial.println(activations[i]);
    }*/
    
    // now we still need to add the bias of this layer to the already calculated activations
    for (int i = 0; i < myModel->layers[layerCounter].numNeurons; i++) {
      activations[i] = activations[i] + pgm_read_float(&myModel->layers[layerCounter].bias[i]);
    }

    // when the last layer is reached, return the activations
    if (layerCounter == myModel->numLayers - 1) {
      return activations;
    }

    // apply activation function (HERE because not for the last layer, it follows softmax which means we dont do anything)
    relu(activations, myModel->layers[layerCounter].numNeurons);

    //set numInputs for next layer -> The number of inputs of the next layer is the number of neurons in THIS layer
    numInputs = myModel->layers[layerCounter].numNeurons;
    
    //set inputs for next layer -> the inputs for the next layer is the activation of THIS layer
    for (int i = 0; i < myModel->layers[layerCounter].numNeurons; i++) {
      input[i] = activations[i];
      //Serial.println(input[i]);
      //Serial.println(activations[i]);

    }
    //Serial.println("unten");
  }
}

/**
 * Apply the activationfunction relu on a pointer (so void as return value)
 */
void relu(double* input,int number){
  for(int i = 0; i < number; i++){
    if(input[i]<0){
      input[i] = 0;
    }
  }
  return input;
}
