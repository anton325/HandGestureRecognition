"""
UseAllBits
this is a file directed at saving even more space. The max values of the label/indptr arrays are often
below a certain value. eg < 256. 256 can be represented with 8 bit, so we save 2 numbers in a 2 byte integer
because thats how much space is reserved in an arduino filesystem

This function is supposed to be used by the convertToC... files when an array gets printed 
to the .h file in the "writeContent" function and replaces the "unpackArray" function
It should return a string with comma seperated weights, where one weight is actually two

Author: Anton Giese
Date: 26.10.2020
"""

import numpy as np

"""
processArray
@param weights: the np weight array to get processed to a string
@param bit: The number of bits one weight is supposed to take
DISCLAIMER: Only 4 is supported right now. 16 is always an option by just declaring the 
array to be of type int (in the c code) 
8 is also built in with the "byte" identifier
Here I attempt to force 2 numbers in one byte
"""

def processArray(weights,bit):
    bytelength = 8  # number of bits in a byte
    string = ""
    weightcounter = 0
    twoWeights = 0
    for weight in weights:
        print("start: ",twoWeights)
        # find out if we need to store it in the first or last 4 bit
        if weightcounter%int((bytelength/bit)) == 0:
            # in the first 4
            twoWeights += weight
            #print(twoWeights)
            # shift it 4 to the right
            twoWeights = twoWeights << (int(bytelength/bit)*2)
            print("first 4: ",twoWeights)
            

        else:
            # in the last 4
            twoWeights+= weight
            string = string + str(twoWeights) +","
            twoWeights = 0
        
        weightcounter+=1

    string = string[0:len(string)-1]
    return string



if __name__ == "__main__":
    a = np.array([1,2,3,4,5,6,7,8,9,10])
    print(processArray(a,4))

