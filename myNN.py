"""
myNN
Purpose of this script:

Execute a neural network that was constructed using Keras without using keras

Here we do not use the built in keras functions but our own functions to evaluate a network.
Here its important that the keras model must only consist of dense (normal) layers

To that purpose we load a keras .h5 model, extract the weights and multiply them accordingly with the input

The evaluatePerFrame function iterates over the train and test data and prints all the gestures
that the network is unable to classify correctly to a file 

Disclaimer: It only supports Relu networks where the last layer is softmax (executed as argmax)

Author: Anton Giese
Date: 26.10.2020
"""

import keras
from keras.models import load_model
import numpy as np
import time as t
import tensorflow_model_optimization as tfmot

import matplotlib.pyplot as plt
from gatherGestures import gatherGestures
import csv


class myNN():
    """
    init
    saves the model
    compiles the model (unnecessary, only neccessary for keras application)
    extracts and saves the weights
    apply csc on the weights matrix
    """
    def __init__(self,model,name):
        self.model = model
        self.name = name
        self.compile()
        #try:
        self.extractWeights()
        #except IndexError:
        #print("IndexError during init of the myNN class")




# ------------------------------------------------------ COLLECT AVAILABLE AND NECESSARY INFORMATION FROM THE MODEL -------------------------------------------- #
    """
    extractWeights
    Gather the weights and the bias from the model and save them using this class
    Iterate over layers and add according to layer
    """
    def extractWeights(self):
        # get number of layers
        self.numberOfLayers = 0
        for a in range(len(self.model.layers)):
            self.numberOfLayers += 1

        self.weights = np.zeros((self.numberOfLayers), dtype = object)
        self.bias = np.zeros((self.numberOfLayers), dtype = object)
        self.numberOfNeurons = np.array([])

        for layerindex in range(len(self.model.layers)):
            weightsAndBias = self.model.layers[layerindex].get_weights()
            #print(len(self.model.layers))
            self.weights[layerindex] = weightsAndBias[0]
            #for w in self.weights:
            #    print(w)
            #print("Die shape der Weights ist: {}".format(self.weights.shape))
            self.bias[layerindex] = weightsAndBias[1]
            # extract further information
            #self.getNumberOfNeurons(self.weights,self.bias)
            self.numberOfNeurons = np.append(self.numberOfNeurons, weightsAndBias[0].shape[1])
    
    """
    getNumberOfNeurons
    This function determines the number of neurons using the weight matrix
    In the first row of the weight matrix are the weights for the FIRST input
    in the second row for the second input and so on
    This means the number of elements in each row is the number of neurons in the next (and only) layer

    This can be checked using the number of bias. There is one bias for each neuron

    And the number of rows is the number of inputs
    
    """
    def getNumberOfNeurons(self,weights,bias):
        self.numInputs = weights.shape[0]
        if weights.shape[1] == bias.shape[0]:
            self.numNeurons = weights.shape[1]
        print("In this layer we have: "+str(self.numNeurons)+" neurons")
        print("This layer recieves "+str(self.numInputs)+" input values")


# ------------------------------------------------------ EXECUTE THE MODEL -------------------------------------------- #

    """
    think 
    This function accepts one or more inputs. For each singular input it employs the calculateActivationOfEachNeuron function to
    calculate the activation. 
    If there are more than 1 input the function iterates through the inputs and calls calculateActivationOfEachNeuron for each
    singluar input

    The activation is then evaluated using the interpret function. The highest value is the detected number

    @param inputs: the input the function thinks about
    @param methodInt: 0 = normal, 1 = csc, 2 = csc with saving multiplication
    """
    def think(self,inputs): 
        # we just have one input
        activation = self.calculateActivationOfOutputLayer(inputs)
        interpreted = self.interpret(activation)
        return activation,interpreted

    """
    calculateActivationOfOutputLayer
    This function calculates the activation of each neuron using the weights and the input
    Method: Normal
    """
    def calculateActivationOfOutputLayer(self,inputs):
        # flatten input, because the input is of shape (28,28) and we want (784,)
        inputs = inputs.flatten()
        
        for layercounter in range(self.numberOfLayers):
            activations = np.zeros((int(self.numberOfNeurons[layercounter]))) # the array with the activation of each neuron
            #activations = inputs.dot(self.weights[layercounter])# -> THE SHORT WAY, NOT IMPLEMENTED ON MICROPROCESSOR
            #print(activations)
            
            for inputindex in range(len(inputs)):
                for weightindex in range(len(self.weights[layercounter][inputindex])):
                    activation = inputs[inputindex] * self.weights[layercounter][inputindex][weightindex]
                    activations[weightindex] += activation
            
            
            # in the end add bias
            activations += self.bias[layercounter]
            # apply activation (relu) for all layers except output layer
            if layercounter != self.numberOfLayers-1:
                activations = self.activation(activations)

            # for next layer, activation is input
            inputs = activations
        return activations

# ------------------------------------------------------ EVALUATE THE MODEL -------------------------------------------- #
    """
    evaluate
    Check the models accuracy
    """
    def evaluate(self,x,y):
            # if it fails just use the built in keras evaluate function
        self.model.evaluate(x,y)


    """
    evaluatePerFrame
    In contrary to the evaluate function that calculates the accuracy for the test and train set, this function is supposed to 
    return the exact gestures that were not recognized and write them in a file so that
    the user can see exactly which gestures fail
    It prints those unrecognized gestures in a file and prints what he thought they were in the 10th column and in the 11th what it should have been
    """
    def evaluatePerFrame(self):
        f = open("myNNOutput/failedGestureTrain.csv","w")
        writer = csv.writer(f)
        wrongs = 0

        g = gatherGestures()
        fails = np.array([])
        trainx,trainy,testx,testy = g.collectAllGestures()
        for gestureCounter in range(len(trainx)):
            _,interpreted = self.think(trainx[gestureCounter])

            # check if interpretation is correct
            if interpreted != trainy[gestureCounter]:
                #print("wrong")
                wrongs+=1
                # keep track of index of failing gestrues
                fails = np.append(fails,gestureCounter)

                # write gesture to file
                for frameindex in range(0,180,9):
                    row = (trainx[gestureCounter][frameindex:frameindex+9]*1024).astype(int)
                    row = np.append(row,int(interpreted))
                    row = np.append(row,int(trainy[gestureCounter]))
                    writer.writerow(row)

        # repeat for test scenes
        f = open("myNNOutput/failedGestureTest.csv","w")
        writer = csv.writer(f)      
        for gestureCounter in range(len(testx)):
            _,interpreted = self.think(testx[gestureCounter])
            #print(interpreted)
            #print(testy[gestureCounter])
            if interpreted != testy[gestureCounter]:
                wrongs+=1
                fails = np.append(fails,gestureCounter)
                for frameindex in range(0,180,9):
                    row = (testx[gestureCounter][frameindex:frameindex+9]*1024).astype(int)
                    row = np.append(row,int(interpreted))
                    row = np.append(row,int(testy[gestureCounter]))
                    writer.writerow(row)

        # feedback
        print("wrong: ",wrongs/(len(testx)+len(trainx)))            

    



# ------------------------------- CHECK QUANTIZATION ------------------------ #
    """
    numberOfClusters 
    This is a function determined for not yet supported models (quantized with keras)
    Here we iterate through the weights and see how many clusters we have
    """
    def numberOfClusters(self):
        cluster = np.array([])

        # one way for the quantized model
        if self.name == "quantized":
            for row in self.model.get_weights()[5]:
                for w in row:
                    if w not in cluster:
                        cluster = np.append(cluster,w)        
        else:
            for w in self.weightsData:
                if w not in cluster:
                    cluster = np.append(cluster,w)
        
        print("Number of different weights in {} network: {}".format(self.name,len(cluster)))


    """
    showDistOfWeights
    This function is supposed to show how the weights in the weights matrix are distributed. 
    It plots the weights vs. the occurences (Histogramm)
    """
    def showDistOfWeights(self):
        plt.hist(self.weightsData, bins=np.arange(self.weightsData.min(), self.weightsData.max()+1,0.01))
        plt.show()





# ------------------------------------ DEBUG -------------------------------------- #

    """
    showWeights
    This function iterates through the layers of the model
    For each layer it prints the weights and the bias

    If the showAll boolean is true, it shows every single weight
    """
    def showWeights(self,showAll):
        for layer,layerCounter in zip(self.model.layers,range(len(self.model.layers))):
            try:
                weights = layer.get_weights()[0]
                bias = layer.get_weights()[1]
                print("In Layer: {}".format(layerCounter))
                print("Weights: ")
                if showAll == True:
                    for w in weights:
                        print(w)
                else:
                    print(weights)
                print("bias: ")
                if showAll == True:
                    for b in bias:
                        print(b)
                else:
                    print(bias)
            except IndexError:
                print("Layer has no weights or something")

    



# ------------------------------------------------------ HELPER FUNCTIONS -------------------------------------------- #
    """
    actiavtion
    Apply the activation function
    """
    def activation(self,x):
        # linear
        #return x
        #relu
        for index in range(len(x)):
            if x[index]<0:
                x[index] = 0
        return x


    """
    Interpret
    This function basicially finds out, which value in the result array is highest and returns the index
    This is used to analyse the activations of the neurons. The neuron with the highest activation is considered to be 
    the answer of the network to the input picture
    """
    def interpret(self,results):
        max = -10000
        maxIndex = 0
        for index in range(len(results)):
            if results[index] > max:
                maxIndex = index
                max = results[index]
        return maxIndex




    """
    Compile
    Since the model was loaded from a file we need to recompile the model before we can use it
    This is only necessary if we want to use the built in keras functions like predict_classes, evaluate ...
    """
    def compile(self):
        self.model.compile(optimizer = 'adam',
                        loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                        metrics = ['accuracy'])
        print("Model successfully compiled")




if __name__ == "__main__":
    m3 = keras.models.load_model("finalModels/quantizedModelfinal1.h5")
    networkQuantized = myNN(m3,"quantized")
    g = gatherGestures()
    trainx,trainy,testx,testy = g.collectAllGestures()
    
    
    networkQuantized.evaluatePerFrame()
    networkQuantized.evaluate(testx,testy)
    networkQuantized.evaluate(trainx,trainy)