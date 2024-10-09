"""
quantize
Quantize recieves a model as input
Its task is to iterate through the layers and put toghether all the weights
When that array is completed, the kmeans clustering algorithm will cluster those and 
return the centroids and the labels
Then we have to split them and sort them back to the according layers

This file is divided in two parts
In the first half the class quantize quantizes all the weights of the model at once

In the second half there is a class that quantizes the weights of each single layer seperately
-> this is the recommended approach

Author: Anton Giese
Date: 26.10.2020
"""

from convert.sparseMatrix import compressedSparseColums
import keras
from kmeans import myKmeans
import numpy as np
from gatherGestures import gatherGestures
from Testbench import Testbench


class quantize():
    """
    initialize the class
    save the model
    Calculate the number of layers
    """
    def __init__(self,model):
        self.model = model
        self.numLayers = 0
        # calculate number of layers
        for layer in self.model.layers:
            self.numLayers += 1
        #print("number of layers: ",self.numLayers)

    
    """
    saveShapeOfLayerWeights
    If we want to untangle the weights we get back from the kmeans we need to know the original
    shapes of the weights in each layer
    """
    def saveShapeOfLayerWeights(self):
        self.originalShapes = np.array([])
        for layer in self.model.layers:
            #print(layer.get_weights()[0].shape)
            self.originalShapes = np.append(self.originalShapes,layer.get_weights()[0].shape[0])
            self.originalShapes = np.append(self.originalShapes,layer.get_weights()[0].shape[1])
        self.originalShapes = np.reshape(self.originalShapes,(self.numLayers,2))
        #print(self.originalShapes)

    
    """
    cscOfEachLayer
    Apply the csc format on the weights of each layer
    Then put all the non zero entries in one array 
    Furthermore we determine the number of non zero values which is handy for 
    going back from the array with all the weights in it
    This array is then clustered by kmeans
    """
    def cscOfEachLayer(self):
        self.csc = np.zeros((self.numLayers,3),dtype = object) # array to save the 3 csc arrays for each layer
        self.allData = np.array([]) # the data arrays from all layers
        #self.numberOfNonZeroArgs = np.array([]) # number values in data array, per layer

        # iterate through layers
        for layerCounter in range(self.numLayers):
            #print("layer ",layerCounter+1)

            # get weights
            weights = self.model.layers[layerCounter].get_weights()[0]

            #get csc object
            csc = compressedSparseColums(weights)

            # get the 3 csc arrays and save them
            data,indices,indptr = csc.compressMatrix(False) # verbose false
            #self.numberOfNonZeroArgs = np.append(self.numberOfNonZeroArgs,len(data))
            self.csc[layerCounter][0] = data
            self.csc[layerCounter][1] = indices
            self.csc[layerCounter][2] = indptr

            # append data to all data
            self.allData = np.append(self.allData,data)
            #print(data)
            #print(indices)
            #print(indptr)
        #print("All data: ",self.csc)
        #print("end all data")

        return self.allData

    """
    clusterWeights
    We cluster the array with all the weights using kmeans
    It returns the centroids the labels and the inertia and the quantizedWeights
    The quantizedWeights array is basically the old allData array but with the new weights
    We can use that to apply the weights back on the model
    """

    def clusterWeights(self,k):
        myK = myKmeans(self.allData)
        #myK.showDistOfWeights(self.allData)
        self.centroids,self.labels,self.inertia,self.quantizedWeights = myK.cluster(k)
        #myK.showDistOfWeights(self.centroids)
        return self.centroids,self.labels,self.inertia,self.quantizedWeights

    
    """
    applyWeightsOnLayers
    After the clustering we have to connect the new weights with the old weights
    We can use the self.quantizedWeights array because it contains all the weights 
    and put them back on the layers by converting back from csc to a normal matrix

    We can use the decompress matrix function from sparseMatrix
    """
    def applyWeightsOnLayers(self):
        #print(len(self.quantizedWeights))
        #print(len(self.allData))

        # get csc empty object we shall use
        csc = compressedSparseColums(np.array([]))

        # index stating how far we have processed in the array with the weights from ALL layers
        quantizedWeightsIndex = 0

        for layerIndex in range(self.numLayers):
            # get OLD csc arrays for this layer
            # only data is old the rest is still accurate
            oldData = self.csc[layerIndex][0]
            indices = self.csc[layerIndex][1]
            indptr = self.csc[layerIndex][2]
            
            # finding out how many elements to take from the new weights
            lenOldData = len(oldData)
            #print(lenOldData)
            #print(quantizedWeightsIndex)
            #print(len(self.quantizedWeights))
            
            # take the right new weights
            newData = self.quantizedWeights[quantizedWeightsIndex:quantizedWeightsIndex+lenOldData]

            # proceed through the array, save where to start next layer
            quantizedWeightsIndex += lenOldData
            #print(newData)

            # get the weight matrix
            weightMatrix = csc.decompressMatrix(newData,indices,indptr)
            
            # to introduce new weights we also have to introduce the (unchanged) bias
            weightsAndBias = []
            bias = self.model.layers[layerIndex].get_weights()[1]
            newWeights = np.copy(weightMatrix)
            weightsAndBias.append(newWeights)
            weightsAndBias.append(bias)

            works = False
            addedRows = 0
            while not works:
                try:
                    self.model.layers[layerIndex].set_weights(weightsAndBias)
                    works = True
                except:
                    addedRows += 1
                    newWeights = np.zeros((weightMatrix.shape[0]+addedRows,weightMatrix.shape[1]))
                    for rowindex in range(weightMatrix.shape[0]):
                        if rowindex>=weightMatrix.shape[0]:
                            for columnindex in range(weightMatrix.shape[1]):
                                newWeights[rowindex][columnindex] = 0
                        else:
                            for columnindex in range(weightMatrix.shape[1]):
                                newWeights[rowindex][columnindex] = weightMatrix[rowindex][columnindex]

                    weightsAndBias = []
                    weightsAndBias.append(newWeights)
                    weightsAndBias.append(bias)
                
        t1 = Testbench(self.model,"after quant")
        t1.checkAll()
        return self.model


    """
    retrain
    To retrain a clustered model, we need to undertake some special steps
    First we need to retrain the model using the keras function fit
    Then we can calculate the gradient for each weight by substracting 
    the old from the new weight
    We can then add all gradients belonging to the same centroid and thus calculate the new centroid
    """
    def retrainOnce(self,trainX,trainY,learnRate,epochs):
        # save the old model with all the weights
        oldModel = keras.models.clone_model(self.model)
        oldModel.set_weights(self.model.get_weights())


        # array where we add up the gradients (sorted by the index of the centroid)
        gradients = np.zeros((self.centroids.shape))

        # fit the new model, and get new weights
        self.model.fit(trainX,trainY,epochs = epochs, shuffle = True)

        # compare new and old model weights to get the gradients
        # then add them up in the right field of the gradients array
        for layerIndex in range(len(self.model.layers)):
            # get weights of this layer, ignore the bias
            weightsOld = oldModel.layers[layerIndex].get_weights()[0]
            weightsNew = self.model.layers[layerIndex].get_weights()[0]
            #print(weightsOld)

            # iterate through the rows
            for rowIndex in range(weightsOld.shape[0]):
                # iterate through the single values in the rows
                for valueIndex in range(len(weightsOld[rowIndex])):
                    if weightsOld[rowIndex][valueIndex] == 0:
                        # ignore all the values where the old value was 0, we dont touch pruned values
                        #print("new value would have been: ",weightsNew[rowIndex][valueIndex])
                        continue
                    else:
                        #print("Old value: ",weightsOld[rowIndex][valueIndex])
                        #print("New value: ",weightsNew[rowIndex][valueIndex])
                        
                        # calculate the gradient for these two weights
                        grad = weightsNew[rowIndex][valueIndex]-weightsOld[rowIndex][valueIndex]
                        #print("calcualted gradient: ",grad)

                        # find out which index the centroid has
                        # compare with the oldWeight, because in the old model
                        # we only used the centroid values
                        for centroidIndex in range(len(self.centroids)):
                            #print("centroid: ",self.centroids[centroidIndex])
                            #print("old data ",weightsOld[rowIndex][valueIndex])
                            
                            # we cannot compare floats on equality
                            if np.abs(self.centroids[centroidIndex][0]-weightsOld[rowIndex][valueIndex])<0.001:
                                # we found the index
                                # add the calcualted gradient to the already existing gradients
                                gradients[centroidIndex] += grad * learnRate
                                #print("found")
                                break
        
        # Adjust the centroids accordingly to the new found gradients
        #print("calulcated gradients: ",gradients)
        #print("old centroids: ",self.centroids)
        for index in range(len(self.centroids)):
            self.centroids[index] += gradients[index]
        #print("new centroids: ",self.centroids)

        # refresh the quantizedWeights array with the new values,
        # because from that value the model gets built in applyWeightsOnLayers()
        for index in range(len(self.quantizedWeights)):
            self.quantizedWeights[index] = self.centroids[self.labels[index]]
        
        # put the new weights on the model
        self.applyWeightsOnLayers()
        return self.model

    
    """
    retrain
    The function uses retrainOnce to retrain a quantized model
    RetrainOnce adjusts the centroids using the gradients between the weights before and after 
    retraining x epochs. Here x cannot be big, because then the gradients become very big very fast
    The finest settings can be reached by applying a single fit epoch
    """
    def retrain(self,trainX,trainY,epoch):
        # define a learn rate
        learnRate = 0.1
        for e in range(epoch):
            # adjust learnrate
            if learnRate>0.05:
                learnRate *= 0.92
            # retrain
            self.retrainOnce(trainX,trainY,learnRate,1)
        return self.model

    """
    Compile
    Since the model was loaded from a file we need to recompile the model before we can use it
    """
    def compile(self):
        self.model.compile(optimizer = 'adam',
                        loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                        metrics = ['accuracy'])
        print("Model successfully compiled")




            

    """
    getQuantizedModel
    Execute all the steps to quantize the model
    @param k: The k in the kmeans -> how many clusters do you want
    """
    def getQuantizedModel(self,k):
        self.saveShapeOfLayerWeights()
        self.cscOfEachLayer()
        self.clusterWeights(k)
        # get the model that can be returned:
        m = self.applyWeightsOnLayers()
        return m


"""
quantizePerLayer class
This class calcualtes centroids per layer and not for all weights in sum
"""
class quantizePerLayer():
    """
    initialize the class
    save the model
    Calculate the number of layers
    """
    def __init__(self,model):
        self.model = model
        self.numLayers = 0
        # calculate number of layers
        for layer in self.model.layers:
            self.numLayers += 1
        #print("number of layers: ",self.numLayers)#

    """
    Compile
    Since the model was loaded from a file we need to recompile the model before we can use it
    """
    def compile(self):
        self.model.compile(optimizer = 'adam',
                        loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                        metrics = ['accuracy'])
        print("Model successfully compiled")



    """
    quantizeModel
    Routine that returns a quantized model
    Utilizes quantizeLayer while iterating over the layers
    """
    def quantizeModel(self,k):
        # array to save the calcualted centroids per layyer
        self.centroidsPerLayer = np.zeros((self.numLayers), dtype = object)
        self.biasPerLayer = np.zeros((self.numLayers), dtype = object)
        # array to save data, indices and indptr
        self.csc = np.zeros((self.numLayers,4),dtype = object)

        # iterate over layers
        for layerCounter in range(len(self.model.layers)):
            self.quantizeLayer(layerCounter,k[layerCounter])

        # in the end return the model
        return self.model

    """
    quantizeLayer
    Get the weights of the layer at layerIndex
    Express them in csc format
    Cluster the data array
    Decompress the csc format to a normal layer
    Call the apply weights function to put them back on the model
    """
    def quantizeLayer(self,layerIndex,k):
        # get weights of this layer
        self.weights = self.model.layers[layerIndex].get_weights()[0]

        # save layerIndex so we can access it in other functions as well
        self.layerIndex = layerIndex

        # get csc class and the 3 arrays
        self.cscClass = compressedSparseColums(self.weights)
        data,indices,indptr = self.cscClass.compressMatrix(False)


        # quantize
        self.centroids,self.labels,self.inertia,self.quantizedWeights = self.clusterWeights(data,k)

        # save csc and quantized values, also save the labels
        self.csc[layerIndex][0] = self.quantizedWeights.flatten()
        self.csc[layerIndex][1] = indices
        self.csc[layerIndex][2] = indptr
        self.csc[layerIndex][3] = self.labels

        self.biasPerLayer[layerIndex] = self.model.layers[layerIndex].get_weights()[1]

        # flatten the centroids so we can access them easier (else we would need [0])
        self.centroidsPerLayer[layerIndex] = self.centroids.flatten()

        #print(self.quantizedWeights)
        #print(indices)
        #print(indptr)
        # get the quantization in matrix format
        decompressedWeightsMatrix = self.cscClass.decompressMatrix(self.quantizedWeights.flatten(),indices,indptr)

        # put the weights back on the model
        self.applyWeightsOnLayers(self.layerIndex,decompressedWeightsMatrix)
        #print(self.model.get_weights())


    """
    clusterWeights
    We cluster the array with all the weights using kmeans
    It returns the centroids the labels and the inertia and the quantizedWeights
    The quantizedWeights array is basically the old allData array but with the new weights
    We can use that to apply the weights back on the model
    """

    def clusterWeights(self,data,k):
        myK = myKmeans(data)
        #myK.showDistOfWeights(self.allData)
        #print("data ",data)
        centroids,labels,inertia,quantizedWeights = myK.cluster(k)
        #print("centroids: ",centroids)
        #myK.showDistOfWeights(self.centroids)
        return centroids,labels,inertia,quantizedWeights



    """
    applyWeightsOnLayer
    Recieves an two dimensional array of weights and simply puts them
    on the appropiate layer
    """
    def applyWeightsOnLayers(self,layerIndex,weights):
        weightsAndBias = []
        # get the bias TAKE OLD OR NEW BIAS??? ANTON says should be new, since they get adjusted during training and they are not objective
        # of quantization because all bias get saved as floats anyways
        bias = self.model.layers[layerIndex].get_weights()[1] 
        #bias=self.biasPerLayer[layerIndex] 
        #put them together
        #print("old shape: ",weights.shape)
        works = False
        # reattach them
        newWeights = np.copy(weights)
        weightsAndBias.append(newWeights)
        weightsAndBias.append(bias)

        addedRows = 0
        while not works:
            try:
                self.model.layers[layerIndex].set_weights(weightsAndBias)
                works = True
                #print("worked")
            
            except:
                #print("did not work")
                # the number of rows might have been changed during the csc process -> this row must have been completely zeros -> add a row of zeros
                #print("add row")
                addedRows += 1
                newWeights = np.zeros((weights.shape[0]+addedRows,weights.shape[1]))
                for rowindex in range(weights.shape[0]):
                    if rowindex>=weights.shape[0]:
                        for columnindex in range(weights.shape[1]):
                            newWeights[rowindex][columnindex] = 0
                    else:
                        for columnindex in range(weights.shape[1]):
                            newWeights[rowindex][columnindex] = weights[rowindex][columnindex]
                #print("new shape: ",newWeights.shape)
                weightsAndBias = []
                weightsAndBias.append(newWeights)
                weightsAndBias.append(bias)

                        





    
    """
    retrain
    To retrain we use the keras fit function for one epoch to calcualte new weights
    To obtain the gradients we utilize the difference between the new and the old weights
    and the learning rate
    """
    def retrain(self,trainX,trainY,epochs):
        learnRate = 0.001
        for e in range(epochs):
            if learnRate>0.0005:
                learnRate *= 0.92
            
            # save the old model
            oldModel = keras.models.clone_model(self.model)
            oldModel.set_weights(self.model.get_weights())

            oldWeights = np.zeros((self.numLayers), dtype = object)
            for layercounter in range(len(oldModel.layers)):
                oldWeights[layercounter] = oldModel.layers[layercounter].get_weights()[0]
                self.biasPerLayer[layercounter] = oldModel.layers[layercounter].get_weights()[1]
            #    print("old weights ",oldModel.layers[layercounter].get_weights()[0])
            #print("end")
            #print(oldWeights)

            
            #oldModelWeights = oldModel.get_weights()
            #print("old model weights")
            #print(oldModelWeights[1])
            #print(oldModel.layers[0].get_weights()[0])
            #if e%10 == 0:
            #    t2 = Testbench(self.model,"quantized")
            #    t2.getTestData()
            #    print("After {} epochs".format(e+1))
            #    t2.checkAll()
            self.model.fit(trainX,trainY,epochs = 1, shuffle = True)
            #if e%10 == 0:
            #    t2 = Testbench(self.model,"quantized")
            #    t2.getTestData()
            #    print("After {} epochs".format(e+1))
            #    t2.checkAll()

            for layerCounter in range(len(self.model.layers)):
                #print("old")
                #print(oldWeights[layerCounter])
                #print("new")
                newWeights = self.model.layers[layerCounter].get_weights()[0]
                #print(newWeights)

                gradients = np.zeros((len(self.centroidsPerLayer[layerCounter]))) # get as many gradients as there are clusters

                #print(self.centroidsPerLayer[layerCounter])

                #print(self.csc[layerCounter][0])

                # calculate the difference between the two models
                for row in range(oldWeights[layerCounter].shape[0]):
                    for column in range(oldWeights[layerCounter].shape[1]):
                        if oldWeights[layerCounter][row][column] == 0:
                            # dont do anything, dont touch pruned weights
                            continue
                        else:
                            grad = oldWeights[layerCounter][row][column]-newWeights[row][column]
                            #print(grad)
                            if(np.abs(grad) > 0.0001):
                                #print("high grad")
                                # find the centroid to which the change belongs
                                for c in range(len(self.centroidsPerLayer[layerCounter])):
                                    #print("centroid: ",self.centroidsPerLayer[layerCounter][c])
                                    #print("old weight: ",oldModelWeights[layerCounter][row][column])
                                    if np.abs(self.centroidsPerLayer[layerCounter][c]-oldWeights[layerCounter][row][column])<0.0001:
                                                # we found the index
                                                # add the calcualted gradient to the already existing gradients
                                                gradients[c] += grad * learnRate
                                                #print("found")
                                                break

                for c in range(len(self.centroidsPerLayer[layerCounter])):
                    self.centroidsPerLayer[layerCounter][c] += gradients[c]

                #print("\nGradients in layer {} : {}".format(layerCounter,gradients))
                #print("new centroids in layer {} : {}".format(layerCounter,self.centroidsPerLayer[layerCounter]))
                #print("old quantized weights ",self.csc[layerCounter][0])
                for index in range(len(self.csc[layerCounter][0])):
                    self.csc[layerCounter][0][index] = self.centroidsPerLayer[layerCounter][self.csc[layerCounter][3][index]]
                #print("new quantized weights ",self.csc[layerCounter][0])

                # decompress matrix
                weightMatrix = self.cscClass.decompressMatrix(self.csc[layerCounter][0],self.csc[layerCounter][1],self.csc[layerCounter][2])

                self.applyWeightsOnLayers(layerCounter,weightMatrix)

                #print("new weights:")
                #for layer in self.model.layers:
                #    print(layer.get_weights()[0])
                #print("end new weights")
            #if e%10 == 0:
            #    t2 = Testbench(self.model,"quantized")
            #    t2.getTestData()
            #    print("After {} epochs".format(e+1))
            #    t2.checkAll()

        return self.model





            


    


# debug purpose: 
if __name__ == "__main__":
    """
    m = keras.models.load_model("prunedModelFinal.h5")
    q = quantize(m)
    q.saveShapeOfLayerWeights()
    q.cscOfEachLayer()
    print(q.clusterWeights(7))
    m2 = q.applyWeightsOnLayers()
    """
    g = gatherGestures()
    trainX,trainY,_,_ = g.collectAllGestures()
    m = keras.models.load_model("finalModels/prunedModelFinal9896.h5")
    m.compile(optimizer = 'adam',
                        loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                        metrics = ['accuracy'])
    q = quantize(m)
    q.getQuantizedModel(12)
    m = q.retrain(trainX,trainY,200)
                            
    t = Testbench(m,"quanti")
    t.getTestData()
    t.checkAll()