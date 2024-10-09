/**
  @file    model.c
  @author  Anton
  @version 0.1
  @date    11.9.2020

  @brief   The implementation of a feedforward neural network
*/

#include "model.h"
#include "model_properties_Full.h"
#include <SoftwareSerial.h>


/**
   public function
   Calls the activation of outputlayer function to get the activation
   Then it checks at which neuron we have the highest value -> the index is the output
*/
int evaluateInput(struct model* myModel, double* input) {
  // reuse input pointer to save memory
   input = activationOfOutputlayer(myModel, input);

  //find index of max neuron
  input[myModel->numOutputs] = -100;      //save maxActivation in input[myModel->numOutputs] to save memory
  int indexOfMax = -1;
  for (int index = 0; index < myModel->numOutputs; index++) {           // iterate through all outputs
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
int thisLabel;

double* activationOfOutputlayer(struct model* myModel, double* input) {
  // save number of Inputs and number of layers
  int numInputs = myModel->numInputs;
  
  //make activations point behind incoming puffer, to save memory
  activations = (input+180);

  //iterate over the layers
  for (int layerCounter = 0; layerCounter < myModel->numInputs; layerCounter++) {
    
    // set all the activations zero
    for (int j = 0; j < myModel->layers[layerCounter].numNeurons; j++) {
      activations[j] = 0;
    }
    int weightCounter = 0;
    int extractedLabel = 0;

    //iterate through weights and inputs
    for (int inputCounter = 0; inputCounter < numInputs; inputCounter++) {
      for (int weightCounterLeavingThisInput = 0; weightCounterLeavingThisInput < myModel->layers[layerCounter].numNeurons; weightCounterLeavingThisInput++) {
        // get weight
        Serial.println(inputCounter * myModel->layers[layerCounter].numNeurons + weightCounterLeavingThisInput);
        //check if label is in first or last 4 bit -> WHEN WE HAVE 16 WEIGHTS OR LESS IT WORKS
        if (weightCounter%2 == 0){
          thisLabel = pgm_read_word(&myModel->layers[layerCounter].labels[inputCounter * myModel->layers[layerCounter].numNeurons + weightCounterLeavingThisInput]);
          extractedLabel = thisLabel >> 4;
        }
        else{
          extractedLabel = thisLabel && 15; // to get the last 4 bits out of the label -> does this work?????
        }
        weight = pgm_read_float(&myModel->layers[layerCounter].centroids[extractedLabel]);
        //Serial.println(weight);
        //check if the weight is zero or not 
        //if(weight > 0.001 || weight < -0.001){        
        double activation = weight * input[inputCounter];         // calculate new activation
        
        activations[weightCounterLeavingThisInput] = activations[weightCounterLeavingThisInput] + activation; // add actiavtion for this neuron to the previous calcualted
        //}
        weightCounter++;
      }
    }
    


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

      
    numInputs = myModel->layers[layerCounter].numNeurons;                         //set numInputs for next layer -> The number of inputs of the next layer is the number of neurons in THIS layer
    
                    
    for (int i = 0; i < myModel->layers[layerCounter].numNeurons; i++) {        //set inputs for next layer -> the inputs for the next layer is the activation of THIS layer
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
