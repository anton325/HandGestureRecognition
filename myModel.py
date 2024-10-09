"""
myModel
MyModel is the brain and heartpiece of creating the model

It uses all the available files to piece together a final model that can be transformed into c code

First it gets all the testing and training data

Then it creates a basis Model using trainBasisModel.py

Then it uses prune.py to prune and retrain

In the end quantization is necessary

After each step the testbench can be uses to check on the changes

Author: Anton Giese
Date: 26.10.2020
"""

import os
import keras
import trainBasisModel
import prune
import Testbench
import numpy as np
import gatherGestures
from quantize import quantize
from quantize import quantizePerLayer

"""
basis
Get the basis model with the configuration 180-8-5 
However by changing the variables this can be customized
"""
def basis(run):
    numLayers = 2 #3
    numNeurons = np.array([8,5])
    activations = np.array(['relu','softmax'])
    epoch = 2000
    model = trainBasisModel.createTrainedModel(numLayers,numNeurons,activations,epoch,False)

    t1 = Testbench.Testbench(model,"basis")
    t1.getTestData()
    #t1.introduceTestData(testX,testY)
    t1.checkAll()
    model.save("myModelOutput/basisModelFinal"+str(run)+".h5")
    return model


"""
pruneModel
Apply pruning to a model
With the specified sparsity
@param sparsity: parameter that will be given to prune function
"""
def pruneModel(model,sparsity):
    epochPrune = 2000
    modelPruned = prune.prune(model,trainX,trainY,testX,testY,epochPrune,sparsity)
    modelPruned.save("myModelOutput/prunedModelFinal.h5")
    return modelPruned


"""
quantizeModelPerLayer
This is the quantize function. 
This is the recommended approach to quantization
@param epoch: How many retraining epochs there should be
@param k: K is an array where each element says how many clusters there should be in the respective layer
"""
def quantizeModelPerLayer(model,k,epoch):
    q = quantizePerLayer(m)
    q.compile()
    m2 = q.quantizeModel(k)
    m2 = q.retrain(trainX,trainY,epoch)
    
    m2.save("myModelOutput/quantizedModelFinal.h5")
    return m2
    
"""
quantizeModelAllInOne
This can quantize ALL weights at once. Dont use this
"""
def quantizeModelAllInOne(model,k,epoch):
    q = quantize(m)
    q.compile()
    m2 = q.getQuantizedModel(k)
    m2 = q.retrain(trainX,trainY,epoch)
    #t3.showWeights()
    m2.save("myModelOutput/quantizedModelFinal.h5")
    #t3.checkAll()




"""
main
"""
if __name__ == "__main__":
    # get data
    g = gatherGestures.gatherGestures()
    trainX,trainY,testX,testY = g.collectAllGestures()
    a=basis(1)
    pruneModel(m,0.5)

    m = keras.models.load_model("finalModels/prunedModelFinal9896.h5")

    # perform initial check on testbench
    t3 = Testbench.Testbench(m,"pruned")
    t3.introduceTestData(trainX,trainY,testX,testY)
    t3.checkAll()

    # get 12 quantized models for different combinations of number of clusters in first and second layer
    for first in range(3,25):
        # first create the folder where all models are savec
        try:
            os.mkdir(os.getcwd()+"/finalModelsEmperically/basisModelsPrunedQuant/"+str(first)+"infirst")
        except:
            print("already exists")
        for second in range(2,9):
            # again create folder in folder
            try:
                os.mkdir(os.getcwd()+"/finalModelsEmperically/basisModelsPrunedQuant/"+str(first)+"infirst/"+str(second)+"insecond")
            except:
                print("already exists")

            # save all results of this configuration in two arrays
            testsetRes = np.array([])
            trainsetRes = np.array([])
            for cycle in range(12):
                # load old model
                m = keras.models.load_model("finalModels/prunedModelFinal9896.h5")                
                print("first ",first)
                print("second ",second)
                # quantize it
                new = quantizeModelPerLayer(m,[first,second],60)
                # check it 
                t3 = Testbench.Testbench(new,"quantized")
                t3.introduceTestData(trainX,trainY,testX,testY)
                t3.getTestData()
                res = t3.checkAll()
                testsetRes = np.append(testsetRes,res[0][1])
                trainsetRes = np.append(trainsetRes,res[1][1])            
                # save it
                new.save("finalModelsEmperically/basisModelsPrunedQuant/"+str(first)+"infirst/"+str(second)+"insecond/"+str(cycle)+".h5")
                print("index: {} for first: {} and second: {}".format(cycle,first,second))
            # save arrays
            np.save("finalModelsEmperically/basisModelsPrunedQuant/"+str(first)+"infirst/"+str(second)+"insecond/testsetResults.npy",testsetRes)
            np.save("finalModelsEmperically/basisModelsPrunedQuant/"+str(first)+"infirst/"+str(second)+"insecond/trainsetResults.npy",trainsetRes)
    

    
