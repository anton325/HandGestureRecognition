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
#include "model_properties_CSC.h"
#include <SoftwareSerial.h>


/* SECTION 2: Private macros                                       */


/* SECTION 3: Private types                                        */


/* SECTION 4: Public variables  :: definitions, no extern
   (must match declarations in header file)                        */


/* SECTION 5: Private variables :: definitions, static mandatory
  (no need to declare, definitions include declarations)           */



/* SECTION 6: Private functions :: declarations, static mandatory
   Rule exception (ISRs)        :: declarations, no static         */


/**
   Private function
   @brief: load the model specified in model_properties.h

*/
//struct model loadModel();


/* SECTION 7: Private functions :: definitions, static mandatory
   Rule exception (ISRs)        :: definitions, no static
   Public functions             :: definitions, no extern
   Function definitions (private & public) written in any order    */

/**
   public function init model
*/
/*
  struct model initModel(){
  struct model myModel = loadModel();
  return myModel;
  }*/


/**
   private function load model, used by the initModel function
*/
/*
  struct model loadModel(){

  struct model* myModel = (struct model *) malloc(sizeof(struct model)); //allocate memory

  // set the values for the model.
  // all the values from the right side are "extern" values, declared and imported from model_properties.h
  myModel->numInputs = numInputs;
  myModel->numOutputs = numOutputs;
  myModel->numLayers = numberOfLayers;


  // malloc for the different layers in the model
  //  myModel->layers = (struct layer*) malloc(myModel->numLayers*sizeof(struct layer));


  // chain the layers togehter with pointers REALLY NECESSARY? PROLLY NOT
  //struct layer* prev = nullptr;

  // iterate through the number of layers and create a layer for each one
  for(int i = 0; i < numberOfLayers; i++){

    // malloc some memory for this layer
    struct layer* thisLayer = (struct layer*) malloc(sizeof(struct layer));


    if(i == 0){
      // if it is the first layer
      thisLayer->numNeurons = numNeuronsLayer1;
      //thisLayer->activationFunction = "leer";
      thisLayer->weights = weightsLayer1;
      thisLayer->bias = biasLayer1;

      // keep the next and prev empty (for now)
      //thisLayer->prev = nullptr;
      //thisLayer->next = nullptr;

      // set prev for next layer
      //prev = thisLayer;

      // save layer in model
      myModel->layers[0] = *thisLayer;
    }

    if(i == 1){
      // if it is the second layer
      thisLayer->numNeurons = numNeuronsLayer2;
      //thisLayer->activationFunction = "leer";
      thisLayer->weights = weightsLayer2;
      thisLayer->bias = biasLayer2;
      // save previous layer
      //thisLayer->prev = prev;
      //thisLayer->next = nullptr;

      //set this layer as next layer in previous layer
      //prev->next = thisLayer;

      // set prev for next layer
      //prev = thisLayer;

      // save layer in model
      myModel->layers[1] = *thisLayer;
    }
  }

  return *myModel;
  }

*/


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
int row;
int currentColumn;

double* activationOfOutputlayer(struct model* myModel, double* input) {
  // save number of Inputs and number of layers
  int numInputs = myModel->numInputs;
  
  //make activations point behind incoming puffer, to save memory
  activations = (input+180);
  
  //int numLayers = myModel->numLayers;


  //debug purpose
  // int myCounter = 0;
  /*double* allValues = (double*) malloc(6*sizeof(double));
    double* allWeights = (double*) malloc(6*sizeof(double));
    double* allIn = (double*) malloc(6*sizeof(double));*/


  //iterate over the layers
  //Serial.println(numLayers);
  for (int layerCounter = 0; layerCounter < myModel->layers; layerCounter++) {

    // get the layer and some more information about it
    //struct layer thisLayer = myModel->layers[layerCounter];
    //int neuronsInThisLayer = thisLayer.numNeurons;

    // get an array for the activations of this layer
    //double* activations = (double*) malloc(neuronsInThisLayer * sizeof(double));

    // set all the activations zero
    for (int j = 0; j < myModel->layers[layerCounter].numNeurons; j++) {
      //Serial.print(activations[j]);
      //Serial.print(" ");
      activations[j] = 0;
      //Serial.print(activations[j]);
      //Serial.print(" ");
    }

    // iterate over the data in this layer
    // keep track of the current column
    currentColumn = 0;

    //Serial.print("number of nnz: ");
    //Serial.println(myModel->layers[layerCounter].numberOfNonZeroValues);
    for(int dataCounter = 0 ; dataCounter < myModel->layers[layerCounter].numberOfNonZeroValues; dataCounter++){
      // check if we are in the next column yet
        //Serial.print("dataCounter: ");
        //Serial.println(dataCounter);
        //Serial.print("Stelle in indexpointer: ");
        //Serial.println(pgm_read_word(&myModel->layers[layerCounter].indptr[currentColumn + 1]));
      while (dataCounter >= pgm_read_word(&myModel->layers[layerCounter].indptr[currentColumn + 1])){
        // if yes, increment
        currentColumn ++ ;
        //Serial.println("column incremented");
        
        //Serial.print("currentColumn: ");
        //Serial.println(currentColumn);
      }
      // check in which row the original maxtrix the non zero value would have been
      row = int(pgm_read_word(&myModel->layers[layerCounter].indices[dataCounter]));
      //Serial.print("Calcualted row: ");
      //Serial.println(row);

      // get the actual weight
      weight = pgm_read_float(&myModel->layers[layerCounter].data[dataCounter]);
      //Serial.print("Calcualted weight: ");
      //Serial.println(weight);

      // add activation
      activations[currentColumn] += input[row] * weight;
      //Serial.print("Calcualted activation: ");
      //Serial.println(activations[currentColumn]);
      
    }
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
