 /** This is an automatically created file by convertToCCSC.py.
  * It was written on 2020-09-28 19:22:04.020355
  * It is the h file of the model module. It contains declarations of the functions and structures for the model
  * The main function provides the input with the 180 features. Using the structs and weights defined in model_propertiesCSC.h
  * it then calcualtes the activations of the neurons. EvaluateInput is able to then return the detected gesture
  * This format does uses optimization. The model has previously been modified and the weights were compressed using CSC
  * Therefore the code for the execution differs compared to the normal model
  */
 
 
 
  #ifndef MODEL_H
  #define MODEL_H
 
 
  #include <stdint.h>
  #include "arduino.h"
 
 
 
    
//  declare  the  layout  for  a  layer
struct  layer{
    int  numNeurons;
    //char*  activationsFunction;
    double*  centroids;
    unsigned  char*  labels;
    unsigned  char*  labelptr;
    unsigned  char*  indices;
    int*  indptr;
    double*  bias;
    int  numberOfNonZeroValues;
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
int  evaluateInput(struct  model*  myModel ,double*  input);
/**
  *  activationOfOutputlayer
  *  takes  a  model  and  returns  the  activation  of  the  last  layer.
  */
double*  activationOfOutputlayer(struct  model*  myModel ,double*  input);
void  relu(double*  input ,  int  number);
    
 
 
#endif //MODEL_H
