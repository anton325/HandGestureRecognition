"""
convertToC
This modul can translate the with keras generated and trained model to the C code
The arduino code has been designed to extract the necessary information from the model_properties.h file
This class` purpose is to take a model and fill the file adquatly 

It writes two files, one the model_properties.h file with the weights and the 
statically initialized structs of the layers and models

And it writes the model.h file, with the declarations of the structs

FURTHER INFORMATION ON HOW TO USE THIS FILE PROPERLY:
First one has to decide on a keras model file to load
Then this class creates a model.h and a model_properties.h file (model_propertiesCSC.h,model_propertiesCSCCluster.h, model_propertiesCSCMult.h)
Those files have to be placed in Arduino/main (Arudino/mainCSC, Arduino/mainCSCCluster, Arduino/mainCSCMult)
In these folders are model.cpp files that support the presented of encoding formats
All other files are identical in the folders


Author: Anton Giese
Date: 26.10.2020
"""


import keras
import numpy as np
import csv
from datetime import datetime
from UseAllBits import processArray ## -> to get the 4 bit compression




class convertToC():
    """
    init
    Save the model
    Save the filename (Always the same)
    Calls create file -> clears existing file
    """
    def __init__(self,model):
        self.model = model
        self.fileNameProperties = "convertOutput/model_properties.h"
        self.fileNameHeader = "convertOutput/model.h"
        # create file
        self.createFiles()
        # already save the information from the model
        self.extractDataFromModel()
        self.calculateTheoreticalSize()


    """
    extractDataFromModel
    This function iterates through the layers and saves weights, bias, number of neurons and general information
    like number of inputs/outputs/layers
    """
    def extractDataFromModel(self):
        # this data does not have to be extracted, its known, but for the sake of it I also 
        # write how it could be extracted
        self.numberOfInputs = 180
        self.numberOfInputs = self.model.layers[0].get_weights()[0].shape[0]

        self.numberOfOutputs = 5
        self.numberOfOutputs = self.model.layers[len(self.model.layers)-1].get_weights()[0].shape[1]

        # init number of layers
        self.numberOfLayers = 0
        #calcualte how many layers there are
        for layer in self.model.layers:
            self.numberOfLayers += 1

        
        self.weightsPerLayer = np.zeros((self.numberOfLayers),dtype=object)
        self.biasPerLayer = np.zeros((self.numberOfLayers),dtype= object)
        self.neuronsPerLayer = np.array([])

        #neuronsLastLayer = 180
        for layerCounter in range(len(self.model.layers)):
            neuronsThisLayer = self.model.layers[layerCounter].get_weights()[0].shape[1]
            self.neuronsPerLayer = np.append(self.neuronsPerLayer, neuronsThisLayer)
            
            weights = self.model.layers[layerCounter].get_weights()[0].flatten()
            self.weightsPerLayer[layerCounter] = weights

            bias = self.model.layers[layerCounter].get_weights()[1]
            self.biasPerLayer[layerCounter] = bias
        
    """
    calculateTheoreticalSize
    This function checks how much size needs saving in this format
    """
    def calculateTheoreticalSize(self):
        numberOfFloats = 0
        for l in self.weightsPerLayer:
            numberOfFloats+= len(l)
        for b in self.biasPerLayer:
            numberOfFloats+=len(b)
        numberOfInts = 0
        for n in self.neuronsPerLayer:
            numberOfInts+=1
        size = numberOfFloats*4+numberOfInts*2
        print("Theoretical size: ",size)
        


    """
    createFile
    There might already exist a file, so we want to clear it instead of appending to it
    """
    def createFiles(self):
        f = open(self.fileNameProperties,"w")
        f.truncate()
        f.close()
        f = open(self.fileNameHeader,"w")
        f.truncate()
        f.close()



    """
    writeHeaderOfPropertiesFile
    Write the header of the properties file, write the basic things that never change
    """

    def writeStartOfPropertiesFile(self):
        f = open(self.fileNameProperties,"a")
        writer = csv.writer(f,delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_NONE,escapechar=',',lineterminator='\n')
        date = str(datetime.now())

        writer.writerow([" /** This is an automatically created file by convertToC.py."])
        writer.writerow(["  * It was written on "+date])
        writer.writerow(["  * This file contains information about the layers in the neural network that is executed"])
        writer.writerow(["  * It utilizes the struct model and struct layer declarations of the model.h file and creates one for each layer and includes the layers in the model"])
        writer.writerow(["  */"])
        writer.writerow([" "])
        writer.writerow([" "])
        writer.writerow([" "])
        writer.writerow(["  #ifndef MODELP_H"])
        writer.writerow(["  #define MODELP_H"])
        writer.writerow([" "])
        writer.writerow([" "])
        writer.writerow(["  #include <stdint.h>"])
        writer.writerow(["  #include <avr/pgmspace.h>"])
        writer.writerow(['  #include "model.h"'])
        writer.writerow([" "])
        writer.writerow([" "])
        writer.writerow([" "])

    """
    writeHeaderOfHeaderFile
    Write the header of the model.h file, write the basic things that never change
    """

    def writeStartOfHeaderFile(self):
        f = open(self.fileNameHeader,"a")
        writer = csv.writer(f,delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_NONE,escapechar=',',lineterminator='\n')
        date = str(datetime.now())

        writer.writerow([" /** This is an automatically created file by convertToC.py."])
        writer.writerow(["  * It was written on "+date])
        writer.writerow(["  * It is the h file of the model module. It contains declarations of the functions and structures for the model"])
        writer.writerow(["  * The main function provides the input with the 180 features. Using the structs defined in model_properties.h"])
        writer.writerow(["  * it then calcualtes the activations of the neurons. EvaluateInput is able to then return the detected gesture"])
        writer.writerow(["  * This format does not use any optimization besides ignoring weights that are zero"])
        writer.writerow(["  */"])
        writer.writerow([" "])
        writer.writerow([" "])
        writer.writerow([" "])
        writer.writerow(["  #ifndef MODEL_H"])
        writer.writerow(["  #define MODEL_H"])
        writer.writerow([" "])
        writer.writerow([" "])
        writer.writerow(["  #include <stdint.h>"])
        #writer.writerow(["  #include <avr/pgmspace.h>"])
        #writer.writerow([' #include "model_p.h"'])
        writer.writerow([' #include "arduino.h"'])


        writer.writerow([" "])
        writer.writerow([" "])
        writer.writerow([" "])


    """
    writeDeclarations
    Responsible for writing strcut declarations
    and function declarations in the header file
    """
    def writeDeclarationsInHeaderFile(self):
        f = open(self.fileNameHeader,"a")
        writer = csv.writer(f,delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_NONE,escapechar=' ',lineterminator='\n')
        writer.writerow(["// declare the layout for a layer"])
        writer.writerow(['struct layer{'])
        writer.writerow(['  int numNeurons;'])
        writer.writerow(['  //char* activationsFunction;'])
        writer.writerow(['  double* weights;'])
        writer.writerow(['  double* bias;'])
        writer.writerow(['};'])

        writer.writerow([' '])
        writer.writerow(['//declare the model containing a number of layers'])
        writer.writerow(['struct model{'])
        writer.writerow(['  int numInputs;'])
        writer.writerow(['  //char* activationsFunction;'])
        writer.writerow(['  int numOutputs;'])
        writer.writerow(['  int numLayers;'])
        writer.writerow(['  struct layer layers['+str(self.numberOfLayers)+'];'])
        writer.writerow(['};'])
        writer.writerow(["  "])
        writer.writerow(["  "])

        writer.writerow([' //defining the public functions'])
        writer.writerow(["int evaluateInput(struct model* myModel, double* input);"])
        writer.writerow(["/**"])
        writer.writerow([" * activationOfOutputlayer"])
        writer.writerow([" * takes a model and returns the activation of the last layer."])
        writer.writerow([" */"])
        writer.writerow(["double* activationOfOutputlayer(struct model* myModel, double* input);"])      
        writer.writerow([" "])
        writer.writerow(['//helper function, used to apply the acitvation function ReLu'])  
        writer.writerow(["void relu(double* input,int number);"])
        writer.writerow(["  "])




    """
    writeContent
    After the header we take care of the content -> the weights/bias/number of neurons in layer. 
    These information do change

    When pasting the weights and bias we need to use the unpack function
    because else it does not have the desired behavior
    This goes in the properties file
    """
    def writeContent(self):
        f = open(self.fileNameProperties,"a")
        writer = csv.writer(f,delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_NONE,escapechar=' ',lineterminator='\n')
        writer.writerow(["//Defining some constants:"])
        writer.writerow(["const int numInputs = "+str(self.numberOfInputs)+";"])
        writer.writerow(["const int numberOfLayers = "+str(self.numberOfLayers)+";"])
        writer.writerow(["const int numOutputs = "+str(self.numberOfOutputs)+";"])

        for layerCounter in range(self.numberOfLayers):
            writer.writerow([" "])
            writer.writerow(["//take care of layer"+str(layerCounter+1)])
            writer.writerow(["const double weightsLayer"+str(layerCounter+1)+" [] PROGMEM= {"+self.unpackArray(self.weightsPerLayer[layerCounter])+"};"])
            writer.writerow(["  "])
            writer.writerow(["const double biasLayer"+str(layerCounter+1)+" [] PROGMEM= {"+self.unpackArray(self.biasPerLayer[layerCounter])+"};"])
            writer.writerow(["  "])
            writer.writerow(["const int numNeuronsLayer"+str(layerCounter+1)+"= "+str(self.neuronsPerLayer[layerCounter])+";"])




    """
    writeStructs
    The function responsible for creating the structs for each layer
    Furthermore it writes the model struct, referencing the created layer structs
    """
    def writeStructs(self):
        f = open(self.fileNameProperties,"a")
        writer = csv.writer(f,delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_NONE,escapechar=' ',lineterminator='\n')
        writer.writerow(["  "])
        writer.writerow(["//Defining the layers used in this model "])
        for layerCounter in range(self.numberOfLayers):
            writer.writerow(["struct layer layer"+str(layerCounter+1)+" = {"])
            writer.writerow(["  .numNeurons = numNeuronsLayer"+str(layerCounter+1)+","])
            writer.writerow(["  .weights = weightsLayer"+str(layerCounter+1)+","])
            writer.writerow(["  .bias = biasLayer"+str(layerCounter+1)])
            writer.writerow(["};"])
            writer.writerow(["  "])
            
        writer.writerow(["//Defining the model and including the just defined layers "])
        writer.writerow(["struct model myModel = {"])
        writer.writerow(["  .numInputs = numInputs,"])
        writer.writerow(["  .numOutputs = numOutputs,"])
        writer.writerow(["  .numLayers = numberOfLayers,"])

        # create the string of how many layers go into the layer field of the model
        layerstring = ""
        for layerCounter in range(self.numberOfLayers):
            layerstring = layerstring + "layer"+str(layerCounter+1)+","
        layerstring = layerstring[0:len(layerstring)-1]

        writer.writerow(["  .layers = {"+layerstring+"}"])
        writer.writerow(["};"])
    """
    unpackArray
    Iterates over array and creates a string consisting of weights seperated with a comma
    Deletes the last false comma 
    """
    def unpackArray(self,array):
        string = ""
        for weight in array:
            string = string + str(weight) +","
        string = string[0:len(string)-1]
        return string
            

    """
    writeEndOfPropertiesFile
    Take care of the end of the properties file -> Again this also is always the same
    """
    def writeEndOfPropertiesFile(self):
        f = open(self.fileNameProperties,"a")
        writer = csv.writer(f,delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_NONE,escapechar=',',lineterminator='\n')
        writer.writerow([" "])
        writer.writerow([" "])
        writer.writerow(["#endif //MODELP_H"])

    """
    writeEndOfHeaderFile
    Take care of the end of the header file -> Again this also is always the same
    """
    def writeEndOfHeaderFile(self):
        f = open(self.fileNameHeader,"a")
        writer = csv.writer(f,delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_NONE,escapechar=',',lineterminator='\n')
        writer.writerow([" "])
        writer.writerow([" "])
        writer.writerow(["#endif //MODEL_H"])






# debug
if __name__ == "__main__":
    # which model to convert
    m = keras.models.load_model("../finalModels/quantizedModelfinal1.h5")
 
         
    # create properties file
    cToC = convertToC(m)
    cToC.writeStartOfPropertiesFile()
    cToC.writeContent()
    cToC.writeStructs()
    cToC.writeEndOfPropertiesFile()

    #write header file
    cToC.writeStartOfHeaderFile()
    cToC.writeDeclarationsInHeaderFile()
    cToC.writeEndOfHeaderFile()