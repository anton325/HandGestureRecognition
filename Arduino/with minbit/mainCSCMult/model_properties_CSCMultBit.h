 /** This is an automatically created file by convertToCCSC.py.
  * It was written on 2020-10-08 16:44:45.103497
  * This file contains information about the layers in the neural network that is executed
  * It utilizes the struct model and struct layer declarations of the model.h file and creates one for each layer and includes the layers in the model
  */
 
 
 
  #ifndef MODELP_H
  #define MODELP_H
 
 
  #include <stdint.h>
  #include <avr/pgmspace.h>
 #include "model.h"
 
 
 
//Defining  some  constants:
const  int  numInputs  =  180;
const  int  numberOfLayers  =  2;
const  int  numOutputs  =  5;
  
//take  care  of  layer1
const  double  centroidsLayer1  []  PROGMEM  =  {-3.0702996253967285 ,-2.4919936656951904 ,-2.1105830669403076 ,-1.8017749786376953 ,-1.4920995235443115 ,-1.2273768186569214 ,-0.9941257238388062 ,-0.7921868562698364 ,-0.37452399730682373 ,0.5625654458999634 ,0.872268557548523 ,1.1558890342712402 ,1.4950588941574097 ,1.835639238357544 ,2.26257061958313 ,2.718916177749634};
const  unsigned  char  labelsLayer1  []  PROGMEM  =  {52.0 ,86.0 ,121.0 ,171.0 ,205.0 ,225.0 ,35.0 ,69.0 ,103.0 ,137.0 ,171.0 ,205.0 ,239.0 ,1.0 ,35.0 ,69.0 ,103.0 ,137.0 ,171.0 ,205.0 ,239.0 ,1.0 ,35.0 ,69.0 ,103.0 ,137.0 ,171.0 ,205.0 ,226.0 ,52.0 ,86.0 ,120.0 ,154.0 ,188.0 ,222.0 ,18.0 ,52.0 ,86.0 ,120.0 ,154.0 ,188.0 ,239.0};
const  unsigned  char  labelptrLayer1  []  PROGMEM  =  {3.0 ,3.0 ,8.0 ,4.0 ,5.0 ,5.0 ,14.0 ,8.0 ,1.0 ,2.0 ,2.0 ,1.0 ,5.0 ,5.0 ,2.0 ,8.0 ,6.0 ,3.0 ,2.0 ,2.0 ,5.0 ,8.0 ,5.0 ,7.0 ,3.0 ,1.0 ,2.0 ,1.0 ,1.0 ,2.0 ,17.0 ,4.0 ,1.0 ,4.0 ,3.0 ,6.0 ,6.0 ,10.0 ,2.0 ,8.0 ,4.0 ,1.0 ,1.0 ,4.0 ,2.0 ,1.0 ,5.0 ,11.0 ,11.0 ,3.0 ,2.0 ,9.0 ,8.0 ,8.0 ,8.0 ,8.0 ,2.0 ,4.0 ,5.0 ,5.0 ,6.0 ,8.0 ,10.0 ,2.0 ,8.0 ,17.0 ,9.0 ,1.0 ,4.0 ,5.0 ,2.0 ,3.0 ,5.0 ,4.0 ,1.0 ,2.0 ,5.0 ,1.0 ,3.0 ,3.0 ,4.0 ,7.0 ,2.0 ,4.0};
const  unsigned  char  indicesLayer1  []  PROGMEM  =  {44.0 ,53.0 ,62.0 ,71.0 ,117.0 ,126.0 ,102.0 ,105.0 ,108.0 ,111.0 ,114.0 ,35.0 ,80.0 ,135.0 ,26.0 ,129.0 ,120.0 ,138.0 ,123.0 ,106.0 ,115.0 ,17.0 ,99.0 ,122.0 ,42.0 ,30.0 ,149.0 ,158.0 ,48.0 ,51.0 ,36.0 ,57.0 ,39.0 ,107.0 ,131.0 ,69.0 ,33.0 ,140.0 ,73.0 ,78.0 ,170.0 ,179.0 ,75.0 ,66.0 ,60.0 ,54.0 ,72.0 ,161.0 ,63.0 ,45.0 ,116.0 ,125.0 ,152.0 ,134.0 ,143.0 ,108.0 ,44.0 ,53.0 ,35.0 ,117.0 ,126.0 ,109.0 ,99.0 ,118.0 ,62.0 ,135.0 ,127.0 ,52.0 ,51.0 ,111.0 ,71.0 ,80.0 ,43.0 ,100.0 ,42.0 ,144.0 ,120.0 ,26.0 ,61.0 ,34.0 ,138.0 ,119.0 ,102.0 ,70.0 ,129.0 ,110.0 ,50.0 ,30.0 ,21.0 ,2.0 ,9.0 ,10.0 ,64.0 ,170.0 ,65.0 ,55.0 ,105.0 ,18.0 ,142.0 ,46.0 ,161.0 ,19.0 ,27.0 ,106.0 ,72.0 ,133.0 ,115.0 ,63.0 ,124.0 ,36.0 ,54.0 ,152.0 ,45.0 ,107.0 ,116.0 ,143.0 ,134.0 ,125.0 ,42.0 ,51.0 ,60.0 ,33.0 ,128.0 ,143.0 ,52.0 ,43.0 ,69.0 ,100.0 ,109.0 ,110.0 ,118.0 ,119.0 ,122.0 ,127.0 ,48.0 ,131.0 ,134.0 ,136.0 ,140.0 ,39.0 ,152.0 ,125.0 ,113.0 ,137.0 ,161.0 ,91.0 ,101.0 ,116.0 ,57.0 ,78.0 ,66.0 ,61.0 ,107.0 ,18.0 ,70.0 ,35.0 ,135.0 ,159.0 ,27.0 ,9.0 ,111.0 ,147.0 ,102.0 ,120.0 ,129.0 ,20.0 ,138.0 ,37.0 ,105.0 ,28.0 ,150.0 ,4.0 ,22.0 ,160.0 ,46.0 ,142.0 ,151.0 ,123.0 ,106.0 ,29.0 ,38.0 ,114.0 ,115.0 ,141.0 ,133.0 ,47.0 ,56.0 ,132.0 ,124.0 ,65.0 ,128.0 ,56.0 ,119.0 ,25.0 ,131.0 ,130.0 ,137.0 ,157.0 ,103.0 ,47.0 ,121.0 ,151.0 ,65.0 ,68.0 ,74.0 ,83.0 ,141.0 ,148.0 ,150.0 ,122.0 ,101.0 ,159.0 ,164.0 ,166.0 ,31.0 ,110.0 ,112.0 ,156.0 ,113.0 ,88.0 ,160.0 ,139.0 ,22.0 ,169.0 ,178.0 ,104.0 ,16.0 ,140.0 ,142.0 ,59.0 ,62.0 ,96.0 ,53.0 ,44.0 ,51.0 ,126.0 ,145.0 ,108.0 ,60.0 ,111.0 ,105.0 ,9.0 ,127.0 ,8.0 ,171.0 ,172.0 ,120.0 ,1.0 ,80.0 ,116.0 ,118.0 ,39.0 ,63.0 ,3.0 ,12.0 ,73.0 ,10.0 ,0.0 ,55.0 ,107.0 ,48.0 ,50.0 ,100.0 ,64.0 ,46.0 ,89.0 ,91.0 ,136.0 ,82.0 ,125.0 ,98.0 ,19.0 ,109.0 ,105.0 ,114.0 ,123.0 ,132.0 ,55.0 ,56.0 ,64.0 ,65.0 ,133.0 ,46.0 ,124.0 ,74.0 ,115.0 ,141.0 ,47.0 ,37.0 ,96.0 ,38.0 ,106.0 ,29.0 ,50.0 ,73.0 ,54.0 ,95.0 ,41.0 ,28.0 ,20.0 ,142.0 ,68.0 ,120.0 ,59.0 ,98.0 ,129.0 ,63.0 ,83.0 ,107.0 ,86.0 ,150.0 ,116.0 ,72.0 ,126.0 ,108.0 ,117.0 ,119.0 ,136.0 ,110.0 ,99.0 ,90.0 ,127.0 ,25.0 ,118.0 ,109.0 ,135.0 ,16.0 ,48.0 ,39.0 ,15.0 ,154.0 ,157.0 ,161.0 ,163.0 ,164.0 ,166.0 ,173.0 ,175.0 ,137.0 ,24.0 ,162.0 ,78.0 ,144.0 ,128.0 ,171.0 ,153.0 ,34.0 ,79.0 ,70.0 ,33.0 ,43.0 ,69.0 ,42.0 ,60.0 ,61.0 ,51.0 ,52.0 ,54.0 ,63.0 ,66.0 ,72.0 ,125.0 ,75.0 ,116.0 ,57.0 ,134.0 ,143.0 ,48.0 ,84.0 ,45.0 ,152.0 ,107.0 ,115.0 ,39.0 ,27.0 ,30.0 ,36.0 ,81.0 ,161.0 ,65.0 ,118.0 ,52.0 ,109.0 ,89.0 ,101.0 ,59.0 ,8.0 ,126.0 ,162.0 ,171.0 ,108.0 ,17.0 ,32.0 ,50.0 ,117.0 ,41.0 ,80.0 ,26.0 ,71.0 ,53.0 ,62.0 ,35.0 ,44.0};
const  int  indptrLayer1  []  PROGMEM  =  {0 ,55 ,55 ,55 ,118 ,190 ,273 ,357 ,403};
const  double  biasLayer1  []  PROGMEM=  {0.4158689 ,-0.015073096 ,-0.013051735 ,0.36578605 ,0.51053727 ,0.0038128397 ,0.48108834 ,0.250305};
const  int  numberOfNonZeroValuesLayer1  =  403;
const  int  numNeuronsLayer1  =  8;
    
  
//take  care  of  layer2
const  double  centroidsLayer2  []  PROGMEM  =  {-18.95896339416504 ,-15.059316635131836 ,-9.497029304504395 ,9.823941230773926};
const  unsigned  char  labelsLayer2  []  PROGMEM  =  {201.0 ,34.0 ,192.0};
const  unsigned  char  labelptrLayer2  []  PROGMEM  =  {1.0 ,1.0 ,1.0 ,2.0 ,1.0 ,2.0 ,1.0 ,1.0 ,1.0};
const  unsigned  char  indicesLayer2  []  PROGMEM  =  {5.0 ,5.0 ,3.0 ,5.0 ,7.0 ,3.0 ,4.0 ,5.0 ,6.0 ,7.0 ,3.0};
const  int  indptrLayer2  []  PROGMEM  =  {0 ,1 ,3 ,5 ,8 ,11};
const  double  biasLayer2  []  PROGMEM=  {-9.359523 ,-1.2569777 ,-1.1336831 ,5.683085 ,2.7815223};
const  int  numberOfNonZeroValuesLayer2  =  11;
const  int  numNeuronsLayer2  =  5;
    
    
//Defining  the  layers  used  in  this  model  
struct  layer  layer1  =  {
    .numNeurons  =  numNeuronsLayer1 ,
    .centroids  =  centroidsLayer1 ,
    .labels  =  labelsLayer1 ,
    .labelptr  =  labelptrLayer1 ,
    .indices  =  indicesLayer1 ,
    .indptr  =  indptrLayer1 ,
    .bias  =  biasLayer1 ,
    .numberOfNonZeroValues  =  numberOfNonZeroValuesLayer1
};
    
struct  layer  layer2  =  {
    .numNeurons  =  numNeuronsLayer2 ,
    .centroids  =  centroidsLayer2 ,
    .labels  =  labelsLayer2 ,
    .labelptr  =  labelptrLayer2 ,
    .indices  =  indicesLayer2 ,
    .indptr  =  indptrLayer2 ,
    .bias  =  biasLayer2 ,
    .numberOfNonZeroValues  =  numberOfNonZeroValuesLayer2
};
    
//Defining  the  model  and  including  the  just  defined  layers  
struct  model  myModel  =  {
    .numInputs  =  numInputs ,
    .numOutputs  =  numOutputs ,
    .numLayers  =  numberOfLayers ,
    .layers  =  {layer1 ,layer2}
};
 
 
#endif
