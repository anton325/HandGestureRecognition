#include "accuracy.h"


double mybuffer[210]; // where we transfer the test data to and scale it

int errors = 0; // number of wrongly classified gestures


void checkAccuracy(struct model* myModel){
  Serial.begin(9600); // enable print
    
  errors = 0;
  // iterate over all the different test gestures
  for(int i = 0; i < numberOfTests;i++){

    // get the start of this scene
    int *startScene = (testset + i*180);

    // copy this scene into buffer and scale it
    for (int j = 0; j < 180; j++){
      mybuffer[j] =  pgm_read_word(&startScene[j])/float(1024);
    }
  
    // time the execution of the network
    unsigned long start = millis();
    int classify = evaluateInput(myModel,mybuffer);
    unsigned long endtime = millis();
    Serial.print("Execution took: ");
    unsigned long t = endtime-start;
    Serial.println(t);
   
   

    // check if network guessed correctly
    int correct = pgm_read_byte(&testlabel[i]);
    if(correct!=classify){
      errors++;
    }
    delay(10);
  }

  // print the findings
  
  Serial.print("Out of ");
  Serial.print(numberOfTests);
  Serial.print(" it got ");
  int rate = numberOfTests-errors;
  Serial.println(numberOfTests-errors);
  /*Serial.print(" correct that is: ");
  float errorrate = errors/float(numberOfTests);
  Serial.print(errorrate);
  Serial.println(" % error rate");*/
  }
