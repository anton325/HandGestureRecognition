"""
convertToCFullLookup
This modul can translate the with keras generated and trained model to the C code
The arduino code has been designed to extract the necessary information from the model_properties_Full.h file
This class` purpose is to take a model and fill the file adquatly 

This is part of the series convertToC, convertToCSC, convertToCCSCMult
The difference in this file is that we dont employ the csc format but still use a small optimization
We have a huge array with the weights. But instead of the weights we only note the labels to
a centroids array we also save

Furthermore we need an adquat model.h file that supports the data which also gets created in this class

This file configures the network in different configurations: 
intOrByte is either "int" "byte" or "bit" and controls the data types

Author: Anton Giese
Date: 26.10.2020

"""


import keras
import numpy as np
import csv
from datetime import datetime
from sparseMatrix import compressedSparseColums





class convertToCCSC():
    """
    init
    Save the model
    Save the filename (Always the same)
    Calls create file -> clears existing file
    """
    def __init__(self,model,intOrByte):
        self.model = model
        if intOrByte == "int":
            self.fileNameProperties = "convertOutput/model_properties_Full.h"
        else:
            self.fileNameProperties = "convertOutput/model_properties_FullByte.h"
        self.fileNameHeader = "convertOutput/model.h"
        self.intOrByte = intOrByte
        # create file
        self.createFiles()
        # already save the items
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

        # create the empty arrays we need to save stuff in later on        
        #self.weightsPerLayer = np.zeros((self.numberOfLayers),dtype=object)
        self.biasPerLayer = np.zeros((self.numberOfLayers),dtype= object)
        #self.neuronsPerLayer = np.array([])


        self.centroidsPerLayer = np.zeros((self.numberOfLayers), dtype = object)
        self.labelsPerLayer = np.zeros((self.numberOfLayers),dtype = object)
       


        #neuronsLastLayer = 180
        for layerCounter in range(len(self.model.layers)):
            #neuronsThisLayer = self.model.layers[layerCounter].get_weights()[0].shape[1]
            #self.neuronsPerLayer = np.append(self.neuronsPerLayer, neuronsThisLayer)
            
            weights = self.model.layers[layerCounter].get_weights()[0]
            mycsc = compressedSparseColums(weights)
            quantizedData = mycsc.quantizeData(weights.flatten())
            self.centroidsPerLayer[layerCounter] = quantizedData[0]
            self.labelsPerLayer[layerCounter] = quantizedData[1]
        

            bias = self.model.layers[layerCounter].get_weights()[1]
            self.biasPerLayer[layerCounter] =bias
        

    def calculateTheoreticalSize(self):
        numberOfFloats = 0
        numberOfInts = 0
        numberOfBytes = 0
        for c in self.centroidsPerLayer:
            numberOfFloats+=len(c)
        for i in self.labelsPerLayer:
            if self.intOrByte == "int":
                numberOfInts += len(i)
            else:
                numberOfBytes+= len(i)

        for b in self.biasPerLayer:
            numberOfFloats+=len(b)
        size = numberOfFloats*4+numberOfInts*2+numberOfBytes*1
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
    writeHeader
    Write the header of the file, write the basic things that never change
    """

    def writeHeaderOfProperties(self):
        f = open(self.fileNameProperties,"a")
        writer = csv.writer(f,delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_NONE,escapechar=',',lineterminator='\n')
        date = str(datetime.now())

        writer.writerow([" /** This is an automatically created file by convertToCFullLookup.py."])
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
        writer.writerow([' #include "model.h"'])
        writer.writerow([" "])
        writer.writerow([" "])
        writer.writerow([" "])


    """
    writeHeaderOfModelFile
    Write the header of the file, write the basic things that never change
    """

    def writeHeaderOfModelFile(self):
        f = open(self.fileNameHeader,"a")
        writer = csv.writer(f,delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_NONE,escapechar=',',lineterminator='\n')
        date = str(datetime.now())

        writer.writerow([" /** This is an automatically created file by convertToCFullLookup.py."])
        writer.writerow(["  * It was written on "+date])
        
        writer.writerow(["  * It is the h file of the model module. It contains declarations of the functions and structures for the model"])
        writer.writerow(["  * The main function provides the input with the 180 features. Using the structs and weights defined in model_propertiesCSC.h"])
        writer.writerow(["  * it then calcualtes the activations of the neurons. EvaluateInput is able to then return the detected gesture"])
        writer.writerow(["  * This format does uses optimization. The model has previously been modified and the weights were compressed using CSC"])
        writer.writerow(["  * Therefore the code for the execution differs compared to the normal model"])
        writer.writerow(["  */"])
        writer.writerow([" "])
        writer.writerow([" "])
        writer.writerow([" "])
        writer.writerow(["  #ifndef MODEL_H"])
        writer.writerow(["  #define MODEL_H"])
        writer.writerow([" "])
        writer.writerow([" "])
        writer.writerow(["  #include <stdint.h>"])
        writer.writerow(['  #include "arduino.h"'])
        #writer.writerow(["  #include <avr/pgmspace.h>"])
        #writer.writerow([' #include "model.h"'])
        writer.writerow([" "])
        writer.writerow([" "])
        writer.writerow([" "])

    
    """
    defineStructs
    Defines the structs in the model header file
    """
    def defineStructs(self):
        f = open(self.fileNameHeader,"a")
        writer = csv.writer(f,delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_NONE,escapechar=' ',lineterminator='\n')
        writer.writerow(["  "])
        writer.writerow(["// declare the layout for a layer"])    
        writer.writerow(['struct layer{'])
        writer.writerow(['  int numNeurons;'])
        writer.writerow(['  //char* activationsFunction;'])
        writer.writerow(['  double* centroids;'])
        if self.intOrByte == "int":
            writer.writerow(['  int* labels;'])
        else:
            writer.writerow(['  unsigned char* labels;'])
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

        writer.writerow(["int evaluateInput(struct model* myModel,double* input);"])
        writer.writerow(["/**"])
        writer.writerow([" * activationOfOutputlayer"])
        writer.writerow([" * takes a model and returns the activation of the last layer."])
        writer.writerow([" */"])
        writer.writerow(["double* activationOfOutputlayer(struct model* myModel,double* input);"])        
        writer.writerow(["void relu(double* input, int number);"])
        writer.writerow(["  "])



    """
    writeContent
    After the header we take care of the content -> the csc matrices/bias/number of nonzerovalues in layer. 
    These information do change

    When pasting the weights and bias we need to use the unpack function because else it does not have the desired behavior
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
            #writer.writerow(["const double weightsLayer"+str(layerCounter+1)+" [] PROGMEM= {"+self.unpackArray(self.weightsPerLayer[layerCounter])+"};"])
            #writer.writerow(["  "])
            writer.writerow(["//take care of layer"+str(layerCounter+1)])
            writer.writerow(["const double centroidsLayer"+str(layerCounter+1)+" [] PROGMEM = {"+self.unpackArray(self.centroidsPerLayer[layerCounter])+"};"])
            if self.intOrByte == "int":
                writer.writerow(["const int labelsLayer"+str(layerCounter+1)+" [] PROGMEM = {"+self.unpackArray(self.labelsPerLayer[layerCounter])+"};"])
            else:
                writer.writerow(["const unsigned char labelsLayer"+str(layerCounter+1)+" [] PROGMEM = {"+self.unpackArray(self.labelsPerLayer[layerCounter])+"};"])
            writer.writerow(["const double biasLayer"+str(layerCounter+1)+" [] PROGMEM= {"+self.unpackArray(self.biasPerLayer[layerCounter])+"};"])
            numNeurons = self.model.layers[layerCounter].get_weights()[0].shape[1]
            writer.writerow(["const int numNeuronsLayer"+str(layerCounter+1)+" = "+str(numNeurons)+";"])
            writer.writerow(["  "])
            
            #writer.writerow(["const int numNeuronsLayer"+str(layerCounter+1)+"= "+str(self.neuronsPerLayer[layerCounter])+";"])




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
            writer.writerow(["  .centroids = centroidsLayer"+str(layerCounter+1)+","])
            writer.writerow(["  .labels = labelsLayer"+str(layerCounter+1)+","])
            writer.writerow(["  .bias = biasLayer"+str(layerCounter+1)+","])
            writer.writerow(["};"])
            writer.writerow(["  "])
            
        writer.writerow(["//Defining the model and including the just defined layers "])

        writer.writerow(["struct model myModel = {"])
        writer.writerow(["  .numInputs = numInputs,"])
        writer.writerow(["  .numOutputs = numOutputs,"])
        writer.writerow(["  .numLayers = numberOfLayers,"])

        layerstring = ""
        for layerCounter in range(self.numberOfLayers):
            layerstring = layerstring + "layer"+str(layerCounter+1)+","
        layerstring = layerstring[0:len(layerstring)-1]

        writer.writerow(["  .layers = {"+layerstring+"}"])
        writer.writerow(["};"])
    """
    unpackArray
    Iterates over array and appends weights with a comma
    Deletes the last false comma 
    """
    def unpackArray(self,array):
        string = ""
        for weight in array:
            string = string + str(weight) +","
        string = string[0:len(string)-1]
        return string
            

    """
    writeEndProperties
    Take care of the end of the file -> Again this also is always the same
    """
    def writeEndProperties(self):
        f = open(self.fileNameProperties,"a")
        writer = csv.writer(f,delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_NONE,escapechar=',',lineterminator='\n')
        writer.writerow([" "])
        writer.writerow([" "])
        writer.writerow(["#endif"])


    """
    writeEndHeader
    Take care of the end of the file -> Again this also is always the same
    """
    def writeEndHeader(self):
        f = open(self.fileNameHeader,"a")
        writer = csv.writer(f,delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_NONE,escapechar=',',lineterminator='\n')
        writer.writerow([" "])
        writer.writerow([" "])
        writer.writerow(["#endif //MODEL_H"])






# debug
if __name__ == "__main__":
    m = keras.models.load_model("../finalModels/quantizedModelfinal1.h5")    
    
    cToC = convertToCCSC(m,"int")
    cToC.writeHeaderOfProperties()
    cToC.writeContent()
    cToC.writeStructs()
    cToC.writeEndProperties()


    cToC.writeHeaderOfModelFile()
    cToC.defineStructs()
    cToC.writeEndHeader()
