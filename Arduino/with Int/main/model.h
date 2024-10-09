 /** This is an automatically created file by convertToC.py.
  * It was written on 2020-10-08 12:09:55.347265
  * It is the h file of the model module. It contains declarations of the functions and structures for the model
  * The main function provides the input with the 180 features. Using the structs defined in model_properties.h
  * it then calcualtes the activations of the neurons. EvaluateInput is able to then return the detected gesture
  * This format does not use any optimization besides ignoring weights that are zero
  */
 
 
 
  #ifndef MODEL_H
  #define MODEL_H
 
 
  #include <stdint.h>
 #include "arduino.h"
 
 
 
//  declare  the  layout  for  a  layer
struct  layer{
    int  numNeurons;
    //char*  activationsFunction;
    double*  weights;
    double*  bias;
};
  
//declare  the  model  containing  a  number  of  layers
struct  model{
    int  numInputs;
    //char*  activationsFunction;
    int  numOutputs;
    int  numLayers;
    struct  layer  layers[2];
};
    
    
  //defining  the  public  functions
int  evaluateInput(struct  model*  myModel ,  double*  input);
/**
  *  activationOfOutputlayer
  *  takes  a  model  and  returns  the  activation  of  the  last  layer.
  */
double*  activationOfOutputlayer(struct  model*  myModel ,  double*  input);
  
//helper  function ,  used  to  apply  the  acitvation  function  ReLu
void  relu(double*  input ,int  number);
    
 
 
#endif //MODEL_H
