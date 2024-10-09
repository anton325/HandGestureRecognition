#include "ANNconstants.h"
#include <avr/pgmspace.h>

// Baud rate used for the serial interface
#define BAUDRATE 115200

// Correct Brightness? 1 = Yes, 0 = No
#define PINHOLE_CAMERA 0

// Lowest and highes output pin used to select a column of the compound eye
#define OUT_PIN_MIN 2
#define OUT_PIN_MAX 4

// Lowest and highes analog input pin used as imput of the compound eye
#define AIN_PIN_MIN A0
#define AIN_PIN_MAX A2

// Lowest and highest digital input pin used to recored buttons that are pressed
#define IN_PIN_MIN  8
#define IN_PIN_MAX 13

// PIN that is inverted after each frame to measure frame rate.
#define PIN_FRAME_RATE 5

// Time in ms to wait per row, that is, between the output pin is set an the analog inputs are read.
#define DELAY_PER_ROW 1 //3

// Number of rows of a frame
#define NO_ROWS (OUT_PIN_MAX - OUT_PIN_MIN + 1)

// Number of rows of a frame
#define NO_COLS (AIN_PIN_MAX - AIN_PIN_MIN + 1)

// Number of pixels of a frame
#define NO_PIXELS (NO_ROWS * NO_COLS)

// Factors to multiply all brightnessValues 
static int brightnessCoorrectionFactors[NO_ROWS][NO_COLS] = {{28, 18, 28},
                                                             {22, 16, 18},
                                                             {28, 18, 28}};
// static int brightnessCoorrectionFactors[NO_ROWS][NO_COLS] = {{16, 16, 16},
//                                                              {16, 16, 16},
//                                                              {16, 16, 16}};

// Value to devide all brightnessValues after multiplying with brightnessCoorrectionFactors.
#define brightnessCoorrectionDivider 16

// Storage for one frame read analog inputs and sent to the serial interface
static int frame[NO_ROWS][NO_COLS]; 

#define PIXELSINFRAME 9

unsigned long time;
unsigned long timeelapsed;
int looptime = 25; // (1/framerate) 25 for 40 fps; 10 for 100 fps

/* 
 * Reads one row and write it to array frame. 
 * The parameter identifies the row. The first row has the value 0. 
 * @param row: row to be read (int)
 */
void readRow(int row) {
  // Activate pin that supplys voltage to the row.
  digitalWrite(row + OUT_PIN_MIN, HIGH);

  // Wait some time. 
  delay(DELAY_PER_ROW);

  // Read add values of the row from the analog input
  for (int col = 0; col < NO_COLS; col++) {
    int brightnessValue = analogRead(col + AIN_PIN_MIN);
    //Serial.print("brightness value: ");
    //Serial.println(brightnessValue);
    frame[row][col] = brightnessValue; 
  }

  // Deactivate pin that supplys voltage to the row.
  digitalWrite(row + OUT_PIN_MIN, LOW);
}

// definitionen ereigniserkennung
#define ALPHA 0.01
#define DELTA 0.01
#define TRESH 0.1

// variablen ereigniserkennung
float tresh_lo;
float tresh_hi;
short mittelwert = 0;
short mittelwert_alt;
float ema = -1;
bool geasture;
int geasturecountdown = 0;

// variablen buffering
int gesturelen;

// recording buffer
#define BUFFSZ 80
int rec_buffer[BUFFSZ][9];
int rec_buffer_ptr = 0;

// scaled buffer
//int scaled_buffer[20*9];
int scaled_buffer_offset = 0;

// ann variables
#define HIDDENLAYERSZ 8 // size of the hidden layer
#define OUTPUTLAYERSZ 5 // size of the output layer
float hiddenlayer[HIDDENLAYERSZ];
float outputlayer[OUTPUTLAYERSZ];


void scale_buffer(){
  int total_bufflen;
  total_bufflen = rec_buffer_ptr;
  //float pseudoindizes[20];

  // decide where to store the scaled values
  if(total_bufflen < BUFFSZ - 20)
    scaled_buffer_offset = BUFFSZ - 20; // at the end of the buffer if there is free memory there
  else
    scaled_buffer_offset = 0; // in place if not
    
  //calculate scaled values
  for(int i=0;i<20;i++){
    int hi,lo;
    float pseudoindex = (float(total_bufflen-1))*float(i)/19; // calculate pseudoindex for interpolation
    lo = floor(pseudoindex);
    hi = ceil(pseudoindex);
    if(lo == hi){ // if the pseudoindex is an integer
      for(int j=0;j<9;j++)
        //scaled_buffer[scaled_buffer_ptr++] = rec_buffer[lo][j]; // just take the value out of the buffer and normalize it
        rec_buffer[i+scaled_buffer_offset][j] = rec_buffer[lo][j]; // just take the value out of the buffer and normalize it
    }
    else{ // if the pseudoindex is between two actual data points
      for(int j=0;j<9;j++){
        float scalingfactor = pseudoindex - lo;
        //scaled_buffer[scaled_buffer_ptr++] = (int)(rec_buffer[lo][j]*(1-scalingfactor) + rec_buffer[hi][j] * scalingfactor); // interpolation and normalization
        rec_buffer[i+scaled_buffer_offset][j] = (int)(rec_buffer[lo][j]*(1-scalingfactor) + rec_buffer[hi][j] * scalingfactor); // interpolation and normalization
      }
    }
  }
  //Serial.println("");
}

void feedidlebuffer(){
  for(short i=0;i<9;i++)
    rec_buffer[rec_buffer_ptr][i] = frame[i/3][i%3];
  rec_buffer_ptr = (rec_buffer_ptr+1) % 3;
}

void switchframes(int* a, int* b){
  for(int i = 0; i<9; i++){
    a[i] = b[i];
  }
}

bool feedrecbuffer(){
  if(rec_buffer_ptr < 3){ // buffering mode changed. rearrange
    int tmp[9];
    if(rec_buffer_ptr == 1){ // rearranging scenario 1
       switchframes(tmp, rec_buffer[0]);
      switchframes(rec_buffer[0], rec_buffer[1]);
      switchframes(rec_buffer[1],rec_buffer[2]);
      switchframes(rec_buffer[2], tmp);
    }
    if(rec_buffer_ptr == 2){ // rearranging scenario 2
      switchframes(tmp, rec_buffer[2]);
      switchframes(rec_buffer[2], rec_buffer[1]);
      switchframes(rec_buffer[1], rec_buffer[0]);
      switchframes(rec_buffer[0],tmp);
    }
    rec_buffer_ptr = 3; // correct pointer for new buffer mode
  }
  for(int i=0;i<9;i++){
    rec_buffer[rec_buffer_ptr][i] = frame[i/3][i%3];
  }
  rec_buffer_ptr++;
  if(rec_buffer_ptr == BUFFSZ){
    init_eventrecog();
    rec_buffer_ptr = 0;
    return false;
  }
  return true;
}

void init_eventrecog() {
  ema = mittelwert;
  //Serial.print("Initializing at ");
  //Serial.println(ema);
  geasture = false;
  geasturecountdown = 0;
  tresh_lo = ema - ema * TRESH;
  tresh_hi = ema + ema * TRESH;
}

bool eventrecog() {
  if(!geasture){
    //feedidlebuffer(); //do the buffering in main!
    if(mittelwert < tresh_lo || mittelwert > tresh_hi){
      geasture = 1;
      return false;
    }
    if(abs(mittelwert-mittelwert_alt) < DELTA){
      //Serial.println("stable");
      ema = ema*(1-ALPHA) + ALPHA*mittelwert;
      tresh_lo = ema - ema * TRESH;
      tresh_hi = ema + ema * TRESH;
    }
  }
  else{
    gesturelen++;
    //feedrecbuffer(); // do the buffering in main
    if(mittelwert > tresh_lo && mittelwert < tresh_hi){
      geasture = 0;
      gesturelen = 0;
      geasturecountdown = 3;
      return false;
    }
    return true;
  }
  return false;
}

int run_ann(void){ //TODO cleanup

  /*
  Serial.print("Debug Buffer:\n");
  for(int i=0;i<9;i++){
    for(int j=0;j<20;j++){
      int t = int(pgm_read_float(&(scaledbuffer[i*20+j]))*1024); //pgm_read_float nötig, um aus PROGMEM zu lesen
      Serial.print(float(t)/float(1024));
      //rec_buffer[j+scaled_buffer_offset][i] = t;
      Serial.print(",");
    }
    Serial.print("\n");
  }
  Serial.print("\n\nActual Buffer:\n");
  for(int i=0;i<9;i++){
    for(int j=0;j<20;j++){
      Serial.print(float(rec_buffer[j+scaled_buffer_offset][i])/float(1024));
      Serial.print(",");
    }
    Serial.print("\n");
  }
  Serial.print("\n\n");*/
  Serial.println(neuralNetwork());
  
}

int neuralNetwork(void){

  // (first) hidden layer
  for(int i = 0;i<HIDDENLAYERSZ;i++){ 
    hiddenlayer[i] = pgm_read_float(&(layer_0_biases[i])); // initialize neuron with bias
  }

  for(int i=0; i<20; i++){  // for each frame
    for(int j=0; j<9;j++){  // for pixel in frame
      float scaledpixel = (float(rec_buffer[i+scaled_buffer_offset][j]) /  float(1024));
      for(int k=0; k < HIDDENLAYERSZ; k++){  // for each neuron
        //hiddenlayer[k] += pgm_read_float(&(layer_0_weights[(i*9+j)*HIDDENLAYERSZ+k])) * scaledpixel;
        hiddenlayer[k] += pgm_read_float(&(layer_0_weights[(i+j*20)*HIDDENLAYERSZ+k])) * scaledpixel;
      }
    }
  }


// apply relu
  for(int i=0;i<HIDDENLAYERSZ;i++){
    if(hiddenlayer[i] < 0)
      hiddenlayer[i] = 0; 
  }
  
  // print hidden layer activations for debug
  /*Serial.println(" ");
  for(int i=0; i < HIDDENLAYERSZ; i++){ Serial.print(hiddenlayer[i]); Serial.print(",");}
  Serial.println(" ");*/

  // output layer
  for(int i=0; i < OUTPUTLAYERSZ; i++){
    // initialize neuron with bias
    outputlayer[i] = pgm_read_float(&(layer_1_biases[i]));
    for(int j=0; j<HIDDENLAYERSZ; j++){
      outputlayer[i] += hiddenlayer[j] * pgm_read_float(&(layer_1_weights[j*OUTPUTLAYERSZ+i]));
    }
    if(outputlayer[i] < 0)
      outputlayer[i] = 0; // apply relu
  }
  // find biggest output value instead of softmax
  int maxout = outputlayer[0];
  int maxoutind = 0;
  int i;
  for(i=0;i<OUTPUTLAYERSZ;i++){
  //  Serial.print(outputlayer[i]);  //TODO cleanup 
   //Serial.print(",");
    if(outputlayer[i] > maxout){
      maxout = outputlayer[i];
      maxoutind = i;
    }
  }
  //Serial.println("");

  return maxoutind;
  
}

/** 
 * Corrects the brightness of the frame of every sensor with factors from brightnessCoorrectionFactors. 
 */
void correctBrightness() {
  for (int row = 0; row < NO_ROWS; row++) {
    for (int col = 0; col < NO_COLS; col++) {
        int brightness = frame[row][col] * brightnessCoorrectionFactors[row][col] / brightnessCoorrectionDivider;
        if (brightness > 1023){
          brightness = 1023;
        }
        frame[row][col] = brightness;
    }
  }  
}

/*
 * readFrame
 * Use readRow to read all the rows
 * Calculate the new mean
 */

void readFrame() {
  // Read all rows 
  for (int row = 0; row < NO_ROWS; row++) {
    readRow(row); 
  }

  if (PINHOLE_CAMERA) {
    correctBrightness();
  }
  mittelwert_alt = mittelwert;
  mittelwert = 0;
  for(int i = 0; i < 9; i++){
    rec_buffer[rec_buffer_ptr][i] = frame[i/3][i%3];
    mittelwert += frame[i/3][i%3]; // i/3 = 0,0,0,1,1,1,2,2,2, integer division
  }
  mittelwert /= 9;
  //Serial.println(mittelwert);
  rec_buffer_ptr++;
}

/*
 * Inverts the pin to measure the frame rate
 */
void invertPinFrameRate() {
  if (digitalRead(PIN_FRAME_RATE) == LOW) {
    digitalWrite(PIN_FRAME_RATE, HIGH);
  } else {
    digitalWrite(PIN_FRAME_RATE, LOW);
  }
}

/* 
 * Writes one frame and the button state to the serial interface. 
 *//*
void writeFrameToSerial() {
  for (int row = 0; row < NO_ROWS; row++) {
    for (int col = 0; col < NO_COLS; col++) {
      Serial.print(frame[row][col]);
      if (row < NO_ROWS - 1 || col < NO_COLS - 1) {
        Serial.print(",");
      } 
    }
  }
  Serial.println(); 
}*/

/*
 * Called once by firmware for initalize firmware.
 */
void setup() {
  Serial.begin(BAUDRATE);


  // Configure output pins to control rows
  for (int i = OUT_PIN_MIN; i <= OUT_PIN_MAX; i++) {
    pinMode(i, OUTPUT);
    digitalWrite(i, LOW); 
  }

  // Configure pin to measure frame rate
  pinMode(PIN_FRAME_RATE, OUTPUT);
  digitalWrite(PIN_FRAME_RATE, LOW); 
}

/*
 * Called by firmware again and again to perform task of software, 
 * that is, to read one frame and write it to the serial interface. 
 */


 /* --------------------------- MAIN LOOP ------------------------ */
void loop() {
  // get time
  time = millis();
  unsigned long start;
  unsigned long end2;
  // Read one picture from the compound eye and write it into the array variable frame.
  readFrame();

  // Write frame and button state to serial interface
  //writeFrameToSerial();
  if(ema == -1)
    init_eventrecog();
  if(!eventrecog()){
    if(geasturecountdown > 0){
      feedrecbuffer();
      geasturecountdown--;
      if(geasturecountdown == 0){
        // event is concluded. evaluate and reset
        //unsigned long dbgtime = millis();
        //Serial.println("Event concluded");
        scale_buffer();
        start = millis();
        run_ann();
        end2 = millis();
        init_eventrecog();
        //Serial.println(millis()-dbgtime);
        //Serial.print("ann time elapsed");
        //Serial.println("plonk");
        //Serial.println(rec_buffer_ptr);
      }
    }
    else{
      feedidlebuffer();
    }
  }
  else{
    feedrecbuffer();
  }

  // Invert pin after each frame to allow measuring the frame rate. 
  invertPinFrameRate();
  timeelapsed = end2-start;
  Serial.print("Elapsed time: ");
  Serial.println(timeelapsed);
  if(timeelapsed < looptime)
    delay(looptime-timeelapsed);
    //Serial.println(timeelapsed);
}
