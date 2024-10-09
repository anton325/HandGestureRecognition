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
#include "model_properties_CSCMultBit.h"
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
unsigned char readOutLabel;

double* activationOfOutputlayer(struct model* myModel, double* input) {
  // save number of Inputs and number of layers
  int numInputs = myModel->numInputs;
  
  //make activations point behind incoming puffer, to save memory
  activations = (input+180);

  //iterate over the layers
  for (int layerCounter = 0; layerCounter < myModel->layers; layerCounter++) {

    // prepare same data for the bit decompression of this layer
    unsigned char valuesInByte;
    unsigned char numBitsOfEachValue;
    if (layerCounter == 0){
      valuesInByte = 2; // in layer one, 2 labels are hidden in each label
      numBitsOfEachValue = 4; // in another way, one label takes up 4 bit of the compressed array
    }
    else{
      valuesInByte = 4; // in label two, 4 labels are hidden in each label, because here we only have 4 centroids
      numBitsOfEachValue = 2; // in another way, one label takes up 4 bit of the compressed array
    }

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
    
    unsigned char labelIndex = 0; // the compressed labels read from PROGMEM
    unsigned char numLabelsRead = valuesInByte; // important for keeping track of how many decompressed labels have already been read from the current compressed label
    unsigned char readLabel;  // the decompressed label
    
    unsigned char labelptrcontent = pgm_read_byte(&myModel->layers[layerCounter].labelptr[labelCount]);
    int indptrcontent = pgm_read_word(&myModel->layers[layerCounter].indptr[currentColumn + 1]);
    
    for(int index = 0 ; index < myModel->layers[layerCounter].numberOfNonZeroValues; index++){
      if(labelsAddedInARow >= labelptrcontent){
        // apply multiplication, all inputs for the same label added up:

        // get decompressed label
        if(numLabelsRead == valuesInByte){ // check if we need to read new compressed label we can decompress
           readOutLabel = pgm_read_byte(&myModel->layers[layerCounter].labels[labelIndex]);
           labelIndex++;  // indicate that we have read another compressed label          
           numLabelsRead = 0; // out of this newly fetched label we have not read one single decomopressed number
        }

        // this is the decompression routine
        readLabel = (readOutLabel >> ((8-(numBitsOfEachValue)-(numBitsOfEachValue)*(numLabelsRead)))) & ((1<<(numBitsOfEachValue))-1);     
        numLabelsRead++; // indicate we have fetched another label out of the compressed

        // now that we know the label, fetch the correct weight
        weight = pgm_read_float(&myModel->layers[layerCounter].centroids[readLabel]);
        activations[currentColumn] += weight* sumInputs;
        labelCount++;
        labelptrcontent = pgm_read_byte(&myModel->layers[layerCounter].labelptr[labelCount]); // update labelptrcontent, since labelCount was iterated
        sumInputs = 0;
        labelsAddedInARow = 0;
      }
      while (index >= indptrcontent){
        // if yes, increment -> we need to go to the next column
        currentColumn ++;
        indptrcontent = pgm_read_word(&myModel->layers[layerCounter].indptr[currentColumn + 1]);
      }
      
      // check in which row the original maxtrix the non zero value would have been
      row = (pgm_read_byte(&myModel->layers[layerCounter].indices[index]));
      sumInputs += input[row];
      labelsAddedInARow++;     
    }


    // at the end it might be possible that we have added up all the inputs but the last multipication has not taken place
    // maybe we need to get another compressed label first
    if(numLabelsRead == valuesInByte){
         readOutLabel = pgm_read_byte(&myModel->layers[layerCounter].labels[labelIndex]);
         labelIndex++;
         numLabelsRead = 0;
      }
      // and decompress it
      // to decompress it we first shift to the right as many times as appropriate
      // then we apply the AND mask to only get the right most bits
      readLabel = (readOutLabel >> ((8-(numBitsOfEachValue)-(numBitsOfEachValue)*(numLabelsRead)))) & ((1<<(numBitsOfEachValue))-1);
      numLabelsRead++;
      activations[currentColumn] += sumInputs * pgm_read_float(&myModel->layers[layerCounter].centroids[readLabel]);

    
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
    }
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
