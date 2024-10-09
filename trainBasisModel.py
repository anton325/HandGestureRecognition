"""
trainBasisModel
This files purpose is to create a keras neural network model, that is able to classify 
gestures

To achive this we use the gatherData module to load all the trainings and evaluation data
Then we create the model and train it

This module has been designed to be used by different classes
In this case the module myModel.py, that controls the creation of the finished model
This is nothing more but a step
Author: Anton Giese
Date: 26.10.2020
"""

import keras
import numpy as np
from gatherGestures import gatherGestures
from keras.utils import np_utils
import matplotlib.pyplot as plt



"""
gatherTheData
employ gatherData to get the trainX,trainY,testX,testY
"""
def gatherTheData():
    g = gatherGestures()
    return(g.collectAllGestures())


"""
get the model
Create a keras sequential model with as many layers and neurons as specified

@param numLayers: How many layers
@param numNeurons: array of size numLayers: how many neurons we want per layer
"""
def getTheModel(numLayers,numNeurons,activations):
    m = keras.models.Sequential()
    for index in range(numLayers):
        if index == 0:
            m.add(keras.layers.Dense(numNeurons[index],input_shape = (180,),activation='relu'))
        else:
            m.add(keras.layers.Dense(numNeurons[index],activation=activations[index]))
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return m

"""
createTrainedModel
One function that returns a trained model, using all the other functions in this module
@param numLayers: Integer, specifying the number of layers
@param numNeurons: Array, stating how many neurons per layer
@param activations: Array, stating which activation function to use
@param epochs: integer, stating how many epochs to perform

The final model gets saved as baseModel.h5

"""
def createTrainedModel(numLayers,numNeurons,activations,epochs,plot):
    # get the model
    model = getTheModel(numLayers,numNeurons,activations)
    model.summary()
    trainX,trainY,testX,testY = gatherTheData()
    #g = gatherData()
    #trainX,trainY = g.loadsyntheticArray()

    trainY = np_utils.to_categorical(trainY)
    testY = np_utils.to_categorical(testY)

    # train:
    history = model.fit(trainX,trainY,epochs = epochs,shuffle = True,batch_size= 5)  
    # evaluate:
    print("Evaluate:")
    model.evaluate(testX,testY)
    # save
    model.save("basisModel.h5")
    return model



# for debug purpose: when this file is main
if __name__ == "__main__":
    numLayers = 2 #3
    numNeurons = np.array([8,5])
    activations = np.array(['relu','softmax'])
    epoch = 1
    createTrainedModel(numLayers,numNeurons,activations,epoch,True)






