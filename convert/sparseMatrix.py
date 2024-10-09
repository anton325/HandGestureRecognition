"""
Sparse matrix is a way to save matrices and save memory space
In this file I am going to present different classes that represent different methods
of compressing matrices
Author: Anton Giese
Date: 26.10.2020
"""
import numpy as np


"""
Compressed Sparse Columns CSC
For further information refer to the work of Naveed, p. 13
This is only supposed to work for two dimensional matrices 

Also you can find the compressMatrixSaveMultiplications function
That is an extension of the normal csc format where the weights get sorted in each column
Then the labelptr array indicates how many weights in a row are the same
This saves even more memory when there are many weights in a row
Also it speeds up execution when many inputs of the neural network can be added and then 
the sum gets multiplied with the respective weight. This is supposed to save a lot of multiplications
"""

class compressedSparseColums():
    """
    init
    Save the matrix 
    Initialize some empty arrays that will later be the 3 smaller arrays that represent the old matrix
    """
    def __init__(self,matrix):
        self.matrix = matrix
        

        

    """
    CompressMatrix
    Iterate over the old matrix and fill the new arrays
    Here it is import to keep track of how many values we have already put into the data array. We do that with dataIndex, Whenever a new row starts, we save dataIndex in the 
    indptr array

    In the end the function checks of what type the arrays can be. If it is possible to save arrays as uint8 it uses less memory
    """
    def compressMatrix(self,verbose):
        self.data = np.array([])
        self.indices = np.array([])
        self.indptr = np.array([0])
        dataIndex = 0   # current location of where we are in the data array right now
        numRows = self.matrix.shape[0] # number of rows in original matrix
        numColumns = self.matrix.shape[1] # number of columns in original matrix

        if verbose:
            print("Anzahl rows in to compress matrix: {}".format(numRows))
            print("Anzahl der columns in to copress matrix: {}".format(numColumns))
        
        for column in range(numColumns):
            for row in range(numRows):
                # for each column go through all the rows
                if self.matrix[row][column] != 0:
                    self.data = np.append(self.data,self.matrix[row][column])
                    dataIndex += 1
                    self.indices = np.append(self.indices,row)
            self.indptr = np.append(self.indptr,dataIndex)
        if verbose:
            print(self.data)
            print(self.indices)
            print(self.indptr)
        
        # cast the arrays to "memory efficient" types
        maxRow = self.getMaxRow()
        numberNonZero = self.numberOfNonZeroValues()
        if verbose:
            print("Highest row with a non zero element: {}".format(maxRow))
            print("Number of non zero elements: {}".format(numberNonZero))
        #self.quantizeData()
        # return the three csc arrays
        return self.data,self.indices,self.indptr

    """
    quantizeData
    When quantization is applied to the csc format, the data array gets split into two.
    The self.kernels array is then the kernels with the few non zero elements left
    The self.labels value is the reference to one of the kernel value that the weight is supposed to be

    First we check if we already have the weight in the kernel, if not append and reference to it
    If yes, we have to find the index of the value in the kernels and reference it

    NOT SURE IF THIS FUNCTION IS ACTUALLY NEEDED! THIS GETS DONE IN QUANTIZE.PY BASICIALLY
    """
    def quantizeData(self,data):
        kernels = np.array([])
        labels = np.array([])
        #data = self.matrix.flatten()
        #print(data)
        for value,valueIndex in zip(data,range(len(data))):
            # check if non zero value is already in the kernel
            if value == 0:
                continue
            if value not in kernels:
                # if not, append it and refernce the value
                kernels = np.append(kernels,value)
                labels = np.append(labels,len(kernels)-1)
            else:
                # when its already in it, we have to reference the value for this index
                for index in range(len(kernels)):
                    if kernels[index] == value:
                        labels = np.append(labels,index)
                        break
        
        maxValue = np.max(labels)
        quantizedData = np.array([kernels,labels])
        #print(self.quantizedData)
        return quantizedData


    """
    compressMatrixSaveMultiplication
    Get the data, indices and indptr like always
    Then sort the data into cluster and sort them based on how high a number is
    Then get labels to the data
    Then sort the labels within the same column and do the same for the indices array
    """
    def compressMatrixSaveMultiplication(self):
        #first get the normal compressed matrix
        dataOld,indices,indptr = self.compressMatrix(False)
        
        # sort the data into cluster
        centroids = np.array([])
        for data in dataOld:
            if data not in centroids:
                centroids = np.append(centroids,data)

        # save the centroids
        centroidsOld = np.copy(centroids)
        #print("centroids: ",centroids)

        # sort the cluster
        centroids = np.sort(centroids)
        #print("new centroids: ",centroids)
        

        # get labels 
        labels = np.zeros((len(dataOld)))
        for dataIndex in range(len(dataOld)):
            for centroidIndex in range(len(centroids)):
                if dataOld[dataIndex] == centroids[centroidIndex]:
                    # matching centroid found, add index of the centroids
                    labels[dataIndex] = centroidIndex
                    break

        # change the order of labels and indices within a single column
        newIndices = np.array([])
        newLabels = np.array([])
        currentColumn = 0

        indicesInThisColumn = np.array([])
        labelsInThisColumn = np.array([])
        for labelIndex in range(len(labels)):
            while labelIndex >= indptr[currentColumn+1]:
                currentColumn += 1
                # we have reached a new column -> sort collected labels and indices
                oldLables = np.copy(labelsInThisColumn)
                labelsInThisColumn, indicesInThisColumn = self.sortTwoArrays(labelsInThisColumn,indicesInThisColumn)
                # append the sorted arrays
                newIndices = np.append(newIndices,indicesInThisColumn)
                newLabels = np.append(newLabels,labelsInThisColumn)

                labelsInThisColumn = np.array([])
                indicesInThisColumn = np.array([])
            labelsInThisColumn = np.append(labelsInThisColumn, labels[labelIndex])
            indicesInThisColumn = np.append(indicesInThisColumn, indices[labelIndex])
        
        # after the loop we have to perform the routine one last time, because it hasnt done that one
        labelsInThisColumn, indicesInThisColumn = self.sortTwoArrays(labelsInThisColumn,indicesInThisColumn)
        newIndices = np.append(newIndices,indicesInThisColumn)
        newLabels = np.append(newLabels,labelsInThisColumn)
        
        # with centroids and labels create the new data
        newData = np.array([])
        for l in newLabels:
            newData = np.append(newData,centroids[int(l)])

        # create the labelptr array and delete duplicates in labelsNew
        #iterate over the labels

        labelptr = np.array([])
        currentColumn = 0
        labelsWithoutDuplicates = np.array([])
        labelsInARow = 0

        for labelindex in range(1,len(newLabels)):
            # check if its still the same column
            justChangedColumns = False
            while labelindex >= indptr[currentColumn + 1]:
                currentColumn += 1
                justChangedColumns = True
            if justChangedColumns:
                labelptr = np.append(labelptr,labelsInARow+1)
                labelsWithoutDuplicates = np.append(labelsWithoutDuplicates,newLabels[labelindex-1])
                labelsInARow = 0    
            else:
                if newLabels[labelindex] == newLabels[labelindex-1]:
                    # its the same label as before (in the same column!!!!)
                    labelsInARow += 1
                    if int(labelindex) == len(newLabels)-1:
                        # its the last index and its the same as before, so add to array, because the loop ends after this iteration
                        labelptr = np.append(labelptr,labelsInARow+1)
                        labelsWithoutDuplicates = np.append(labelsWithoutDuplicates,newLabels[labelindex])
                    
                else:
                    # its not the same as before, or its a different column save it
                    labelptr = np.append(labelptr,labelsInARow+1)
                    labelsWithoutDuplicates = np.append(labelsWithoutDuplicates,newLabels[labelindex-1])
                    if int(labelindex) == len(newLabels)-1:
                        # its the last index and its the same as before, so add to array, because the loop ends after this iteration
                        labelptr = np.append(labelptr,labelsInARow+1)
                        labelsWithoutDuplicates = np.append(labelsWithoutDuplicates,newLabels[labelindex])

                    labelsInARow = 0




        #print("labels /wo duplicates: ",labelsWithoutDuplicates)
        #print("labelptr: ",labelptr)



        return centroids,labelsWithoutDuplicates,labelptr,newIndices,indptr
            

    """
    sortTwoArrays
    This function recieves two arrays of same length
    It sorts the one array by size of the numbers
    And the important part is that it sorts the other array in the same manner, like when
    it changes the numbers with index 1 and 3, it does the same for the other array
    """
    def sortTwoArrays(self,label,indices):
        for index in range(len(label)):
            for index2 in range(len(label)):
                if label[index] < label[index2]:
                    # change them
                    temp = label[index]
                    label[index] = label[index2]
                    label[index2] = temp 
                    #change indices as well
                    temp = indices[index]
                    indices[index] = indices[index2]
                    indices[index2] = temp
        #print(label)
        #print(indices)
        return label,indices


    """
    decompressMatrix
    The inverse function of the compressMatrix function
    We get the data, indices and indptr arrays and return a normal matrix
    """
    def decompressMatrix(self,data,indices,indptr):
        indptrIterator = 0 # iterator over indptr
        dataIterator = 0 # iterator over data
        numColumns = len(indptr) # num of columns
        numRows = np.max(indices) # num rows -> there must be at least this number of rows, there might have been more in the original, but we dont care about them because they must have been completely 0

        # create the orig Array with the right shape -> we need one row more because in indices are only indices saved 
        # and we need one column less because in indptr are always one element more than there are columns
        origArray = np.zeros((int(numRows)+1,int(numColumns)-1))

        # iterate over the indptr
        while indptrIterator < numColumns - 1:
            # check the distance between the two values so that we know how many data values belong in this row
            numDataInThisColumn = indptr[indptrIterator+1]-indptr[indptrIterator]
            upToDataIndex = dataIterator + numDataInThisColumn

            while dataIterator < upToDataIndex:
                #print(indices[dataIterator])
                #print(indptrIterator)
                #print(dataIterator)
                origArray[int(indices[dataIterator])][indptrIterator] = data[dataIterator]
                dataIterator += 1

            indptrIterator += 1 # go to next column
        return origArray


# ------------------------------------------ HELPER FUNCTIONS ------------------------------------- #
    """
    getMaxRow
    This function is designed to find out the highest row with a non zero element. 
    With the result we can determine of which type (int8, int16) the indices array must be
    """
    def getMaxRow(self):
        maxRow = 0
        for row,rowCounter in zip(self.matrix,range(self.matrix.shape[0])):
            for val in row:
                if np.abs(val) > 0.0001:
                    maxRow = rowCounter
        return rowCounter
        
    
    """
    numberOfNonZeroValues
    Calculates the number of values in the matrix that are NOT zero
    """

    def numberOfNonZeroValues(self):
        numberOfNonZero = 0
        for row in self.matrix:
            for value in row:
                if np.abs(value) == 0:
                    numberOfNonZero += 1
        return numberOfNonZero
    
        



# debugging:
# ---------------------------------------- TEST CLASS ------------------------------------- #  
      
if __name__ == "__main__":
    #array = np.array([[0,-0.98,1.48,0],[0,-0.98,2.09,0],[0,0,0,0],[0,0,-0.98,0],[0,0,0,0]])
    array = np.array([[1,-0.5,4],[0,1.5,1.5],[0,4,0],[1,-0.5,4]])
    
    #array = np.array([[0,-0.98,1.48,0],[0,-0.14,-1.08,0],[0,1.92,0,0],[0,0,1.53,0]])

    print(array)
    
    csc = compressedSparseColums(array)
    d,indices,indptr = csc.compressMatrix(True)
    print(csc.compressMatrixSaveMultiplication())
    #d,indices,indptr,_,_ = csc.compressMatrixSaveMultiplication()
    #print(csc.decompressMatrix(d,indices,indptr))





    