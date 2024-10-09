# include "model.h"
# include "buffer.h"
#include "camera.h"



// include the one scene in TestScenes.h
extern const double scene[];
extern struct model myModel;


int frameCount = 0;
Buffer b(0.1, 0.1, 0.1);
//double szene[200];
double thisFrame[9];



void setup() {
  Serial.begin(9600); 
  Serial.println("Initialized");
}


/**
   loop function
   @purpose: The endless loop, everything the controller should repeat all the time comes here
*/
void loop() { 
  /*
  for (int i = 0; i < 9; i++) {
    thisFrame[i] = pgm_read_float(&scene[frameCount*9+i]);
  }
  frameCount++;
  if (frameCount >440){
    Serial.println("Framecount 440");
    frameCount = 0;
  }    

    Serial.println("Frame:");
    for(int a = 0; a < 9; a++){
      //Serial.println(thisFrame[a]);
    }
    */
  readFrame(thisFrame);

  bool r = b.feedFrame(thisFrame);

  
  if(r){
    //Serial.print("Return true at frame");
    //Serial.println(frameCount);

    double* gesture = b.getScaledBuffer();
    Serial.println(" ");
    Serial.println("Evaluate:");
    double start = millis();
    int a = evaluateInput(&myModel,gesture);
    double endPoint = millis();

    Serial.println("This gesture qualifies as: ");
    Serial.println(a);
    double diff = endPoint-start;
    Serial.print("This took: ");
    Serial.println(diff);

  }
  
  delay(5);

}
