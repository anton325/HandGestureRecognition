/**
 * Main
 * Executes a neural network on the arduino
 * 
 * Two different modi:
 * 1: RUN -> capture live feed from the camera and evaluate it using a buffer to reconize the start of a gesture
 * 
 * 2: ACCURACY -> Dont care about the camera and buffer, just send predefined test data through the network
 */


#define RUN
//#define ACCURACY


#ifdef RUN
#include "camera.h"
# include "buffer.h"
#endif
#ifdef ACCURACY
#include "accuracy.h"
#endif

# include "model.h"
// include the one scene in TestScenes.h3
extern struct model myModel;



// CODE FOR RUN
#ifdef RUN
int frameCount = 0;
Buffer b;
int thisFrame[9];

/**
   Function setup:
   @purpose: Gets called once at the start
*/

void setup() {
  Serial.begin(9600); 
  Serial.println("Initialized");
}

/**
   loop function
   @purpose: The endless loop, everything the controller should repeat all the time comes here
*/
void loop() { 
  readFrame(thisFrame);

  bool gestureStart = b.feedFrame(thisFrame);

  
  if(gestureStart){
    double* gesture = b.getScaledBuffer();
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
  delay(5); // necessary to give camera enough time
}

#endif


// CODE FOR ACCURACY
#ifdef ACCURACY
void setup() {
  Serial.begin(9600); 
  Serial.println("Initialized");
}
void loop(){
  checkAccuracy(&myModel);
}
#endif
