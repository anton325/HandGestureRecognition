"""
Testbench model
The testbench is an alround module that tests trained neural networks on different aspects
For once it can gatherData from the gatherData module. Furthermore it evaluates a model on 
test and train data and can check how many weights are zero. Also it can determine the number
of clusters in the weights. 
Some debug functions for printing all the weights can be found as well

author: Anton Giese
Date: 26.10.2020
"""



import keras
import numpy as np
import tensorflow_model_optimization as tfmot
import tensorflow as tf
import gatherGestures




class Testbench():
    """
    init testbench
    copy model and compile it
    Initialize variables, that until now no testdata is saved in the module
    @param model: The model you want to check
    @param name: Just the name of the model to make the output more readable
    """
    def __init__(self,model,name):
        print("\n"*4)
        self.model = model
        self.compile()
        self.testDataAvailable = False
        self.name = name
        print("Testbench successfully initialized")

    def checkAll(self):
        print("Führe CheckAll für "+str(self.name)+" durch")
        if self.testDataAvailable == False:
            self.getTestData()
        res = self.evaluate()
        self.getNumberOfZerosInModel()
        self.getNumberOfClusters()
        return res

    # -------------------------------- PREPARE MODEL ---------------- #

    """
    introduce test data
    This offers a possibility to introduce testdata to the object. 
    On basis of this test data the model will be evaluated
    """
    def introduceTestData(self,trainx,trainy,testx,testy):
        self.testX = testx
        self.testY = testy
        self.trainX = trainx
        self.trainY = trainy
        self.testDataAvailable = True
        print("Test data successfully introduced")

    """
    getTestData
    When using the Testbench for the gesture model, we can use the gatherGestures module
    to load the testData 
    """
    def getTestData(self):
        g = gatherGestures.gatherGestures()
        self.trainX,self.trainY,self.testX,self.testY = g.collectAllGestures()
        self.testDataAvailable = True
       # print(self.testX)


    def checkOnsyntheticData(self):
        g = gatherGestures.gatherGestures()
        x,y = g.loadsyntheticArray()
        self.model.evaluate(x,y)

    

    """
    retrain
    An extra function that retrains the saved model
    """
    def retrain(self,epochs):
        self.model.fit(self.trainX,self.trainY,shuffle = True,epochs = epochs)

    """
    Compile
    Since the model was loaded from a file we need to recompile the model before we can use it
    """
    def compile(self):
        self.model.compile(optimizer = 'adam',
                        loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                        metrics = ['accuracy'])
        print("Model successfully compiled")


    # -------------------------------- CHECK ACCURACY ---------------- #
    """
    evaluate the model
    Using the keras evaluate function the model shall be tested
    This works only if test data was already introduced to this testbench
    """
    def evaluate(self):
        if self.testDataAvailable == False:
            print("Evaluation unsuccessful, data to test on has not yet been introduced")
            return

        print("Evaluate on all test data:")
        resTrainset = self.model.evaluate(self.testX,self.testY,verbose = 1)
        print("Evaluate on all Train Data:")
        resTestset = self.model.evaluate(self.trainX,self.trainY,verbose = 1)
        return resTrainset,resTestset

    # ------------------------------ CHECK RESULT OF PRUNING -------------------- #

    """
    getNumberOfZerosInLayer
    Helper function that gets called with the array of the weights of one layer
    It trys to access the actual weights in the array. If it is of higher dimension it slowly iterates recursivly through the array
    """
    def getNumberOfZerosInLayer(self,weightsLayer):
        numberOfZeros = 0
        totalNumber = 0
        if isinstance(weightsLayer[0],np.ndarray):
            # content is again an array
            for content in weightsLayer:
                deltaNumberOfZeros,deltaTotalNumber = self.getNumberOfZerosInLayer(content)
                numberOfZeros += deltaNumberOfZeros
                totalNumber += deltaTotalNumber
        else:
            for weight in weightsLayer:
                totalNumber += 1
                if np.abs(weight)==0:#x<0.001:
                    numberOfZeros += 1
        return numberOfZeros,totalNumber

    """
    getNumberOfZerosInModel
    To check the effectivity of the pruning we need to find out how many parameters there are in the model
    and how many of those are zero
    To this purpose we iterate through the layers. 
    A helper function then calcualtes the number of zeros in a single layer
    """
    def getNumberOfZerosInModel(self):
        numberOfZeros = 0
        totalNumber = 0
        for layer in self.model.layers:
            try:
                layer.get_weights()[0]
                deltaNumberOfZeros,deltaTotalNumber = self.getNumberOfZerosInLayer(layer.get_weights()[0])
                numberOfZeros += deltaNumberOfZeros
                totalNumber += deltaTotalNumber
            except Exception as e:
                # we arrive here when the layer does not have any weights
                print("Layer flawed")#, message: "+str(e))
                #a = 3
        print("The model has "+str(totalNumber)+" parameters of which are "+str(numberOfZeros)+
         " zeros ({}%)".format(numberOfZeros*100/totalNumber))
        return numberOfZeros,totalNumber


    # ------------------------------ CHECK RESULT OF quantization -------------------- #

    def getNumberOfClusters(self):
        self.cluster = np.array([])
        for layer in self.model.layers:
            try:
                for row in layer.get_weights()[0]:
                    for value in row:
                        if value not in self.cluster:
                            self.cluster = np.append(self.cluster,value)
            except:
                a = 3

        print("The model has {} different weights".format(len(self.cluster)))


    # ------------------------------ DEBUG -------------------- #
    """
    showWeights
    This function iterates through the layers of the model
    For each layer it prints the weights and the bias
    """
    def showWeights(self):
        for layer,layerCounter in zip(self.model.layers,range(len(self.model.layers))):
            try:
                weights = layer.get_weights()[0]
                bias = layer.get_weights()[1]
                print("In Layer: {}".format(layerCounter))
                print("Weights: ")
                print(weights)
                print("bias: ")
                print(bias)
            except IndexError:
                print("Layer has no weights or something")

    """
    predict
    A very simple function to let the network think about the presented input
    """
    def predict(self):
        print(self.model.predict_classes(np.reshape(self.testX[0],(1,-1))))




if __name__ == "__main__":



    # ------------------- LOAD DIFFERENT MODELS ------------------- #
    m4 = keras.models.load_model("finalModels/basisModelFinal9896.h5")
    m5 = keras.models.load_model("finalModels/quantizedModelfinal1.h5")

    # -------------------- TEST TESTBENCH FUNCTIONS ------------------#

    tq = Testbench(m5," final")
    tq.getTestData()
    tq.checkAll()
    print("check on synthetic data")
    tq.checkOnsyntheticData()
    
    
    tq = Testbench(m4,"normal final")
    tq.getTestData()
    tq.checkAll()
    print("check on synthetic data")
    tq.checkOnsyntheticData()
    
