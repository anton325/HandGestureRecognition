"""
#-------------------------- ANNOTATIONS BY ANTON -------------------#
Class PixelFrame:
Purpose: - Get pixel values out of an incoming string that portraits a single FRAME
         - Normalize those values
         - Save within this class
         -> this class IS a Frame



Author: Kubik, with some code extensions by Anton Giese
"""
import numpy as np


class PixelFrame:
    def __init__(self, line):
        self.data = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        #print("line: ",line)
        #print(type(line))
        if type(line) == type([]):
            values = []
            for a in range(len(line)):
                values.append(line[a])

        else:
            values = np.zeros((9))
         #   print("länge line: ",len(line))
            done = False
            doneWithByte = False
            index = 0
            startByte = 0
            #for b in line:
                #print(int(b))
            
            lenByte = 0
            while not done:
                try:
                    lenByte+=1
                    #print(int(line[startByte:startByte+lenByte]))
                    values[index] = int(line[startByte:startByte+lenByte])
                    if index == 8 and lenByte>4:
                        done = True
                        #break
                except:
            #        print("except at byte {} with len {}".format(index,lenByte))
             #       print(values[index])
              #      print(index)
                    index += 1
                    startByte = startByte + lenByte
                    lenByte = 0
            

            
                

                        
        i = 0
        while i < 9: #len(values):
            #print("values: ",float(string(values[i])))
            self.data[i] = float(values[i])/1024
            i = i + 1
        self.mean = sum(self.data) / len(self.data)

    """
    pixeldiff
    compare the single pixels of this frame with another frame and see if it variates or not
    If yes: return 1
    else return 0
    """
    def pixeldiff(self, otherFrame, eps):
        i = 0
        while i < len(self.data):
            # print(str(otherFrame.data[i]-self.data[i]))
            if abs(otherFrame.data[i] - self.data[i]) > eps:
                return 1
            i = i + 1
        return 0
