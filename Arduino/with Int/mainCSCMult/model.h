 /** This is an automatically created file by convertToCCSC.py.
  * It was written on 2020-09-18 14:33:29.636399
  */
 
 
 
  #ifndef MODEL_H
  #define MODEL_H
 
 
  #include <stdint.h>
  #include "arduino.h"
 
 
 
    
struct  layer{
    int  numNeurons;
    //char*  activationsFunction;
    double*  centroids;
    int*  labels;
    int*  labelptr;
    int*  indices;
    int*  indptr;
    double*  bias;
    int  numberOfNonZeroValues;
};
  
struct  model{
    int  numInputs;
    //char*  activationsFunction;
    int  numOutputs;
    int  numLayers;
    struct  layer  layers[2];
};
    
    
int  evaluateInput(struct  model*  myModel ,double*  input);
double*  activationOfOutputlayer(struct  model*  myModel ,double*  input);
void  relu(double*  input ,  int  number);
    
 
 
#endif //MODEL_H
