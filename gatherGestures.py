"""
gatherGestures
This files job is to collect all the csv files with trainings and test data
Using frameBuffer we extract only the gestures from the files 
Also it is possible to print the extracted gestures to file, seperated for test and training -> this means that those files can be watched by the monitor for videos marcus venzke created
It is also capable of loading this one huge Venzke synthetic file which consists of all venzke scenes that were slightly changed (rotated, brightness changed) to create a lot of synthetic data (I think
it was created by Klisch))

This modul is to be used as a helper in different moduls where training and test data is necessary

Author: Anton Giese
Date: 26th of october 2020
"""

import frameBuffer
import csv
from Frame import PixelFrame
import numpy as np


class gatherGestures():
    """
    init
    The init class specifies some files
    """
    def __init__(self):
        #-------------------------------------------- specifiy where the files are -------------------------------------- #
        # the training files from klisch
        self.trainingKlisch = np.array(['data/dataKlisch/MyHighContrast_5cm_Annotated.csv',
                                   'data/dataKlisch/MyLowContrast_5cm_Annotated.csv'])
        self.trainingLabelsKlisch = np.array([[1,1,2,3,4,1,2,3,4,1,2,3,4,],[1,2,4,3,2,1,3,4,2,1,3,4]])


        # the test files from Klisch (They are basically recorded by Kubik. He used 13 files as test set, while Klisch only took the first 8 thats why they are commented out)
        self.testKlisch = np.array(['data/testKlisch/test_fac_highcontrast_3cm-annotated.csv',
                                    'data/testKlisch/test_fac_highcontrast_15cm-annotated.csv',
                                    'data/testKlisch/test_fac_highcontrast_20cm-annotated.csv',
                                    'data/testKlisch/test_fac_highcontrast_30cm-annotated.csv',
                                    'data/testKlisch/test_fac_lowcontrast_3cm-annotated.csv',
                                    'data/testKlisch/test_fac_lowcontrast_15cm-annotated.csv',
                                    'data/testKlisch/test_fac_lowcontrast_20cm-annotated.csv',
                                    'data/testKlisch/test_fac_lowcontrast_30cm-annotated.csv'])
                                    #'data/testKlisch/test_pin_highcontrast_3cm-annotated.csv',
                                    #'data/testKlisch/test_pin_highcontrast_15cm-annotated.csv',
                                    #'data/testKlisch/test_pin_highcontrast_20cm-annotated.csv',
                                    #'data/testKlisch/test_pin_lowcontrast_3cm_006-annotated.csv',
                                    #'data/testKlisch/test_pin_lowcontrast_15cm_004-annotated.csv'])


        # comment to data from Kubik: Kubik has on his CD a folder called trainingsdaten and one called testdaten. Furthermore he has in the folder where he has his code
        # the folder training and test. The folder trainingKubik and testKubik which are found here are copies from the folder where he had his code
        # the training files from Kubik
        self.trainingFilesKubik = np.array(['data/trainingKubik/garbage.csv',
                                       'data/trainingKubik/LR_train_fac_litceil_5-10cm.csv',
                                       'data/trainingKubik/UD_train_fac_litceil_5-10cm.csv',
                                       'data/trainingKubik/LR_train_fac_litceil_30cm.csv',
                                       'data/trainingKubik/UD_train_fac_litceil_30cm.csv',
                                       'data/trainingKubik/UD_train_fac_wall_5-10cm.csv', 
                                       'data/trainingKubik/LR_train_fac_wall_5-10cm.csv', 
                                       'data/trainingKubik/UD_fac_train_dim_various.csv', 
                                       'data/trainingKubik/LR_fac_train_dim_various.csv', 
                                       'data/trainingKubik/UD_pinhole2.csv',
                                       'data/trainingKubik/LR_pinhole2.csv'])
        # The training labels from Kubik
        self.trainingLabelsKubik = np.array([[0],[2,1],[4,3],[2,1],[4,3],[4,3],[2,1],[4,3],[2,1],[4,3],[1,2]])
        # the test files from kubik
        self.testKubik = np.array(['data/testKubik/UDLR2_bright.csv',
                                   'data/testKubik/UDLR2_bright_far.csv',
                                   'data/testKubik/UDLR2_tisch.csv',
                                   'data/testKubik/UDLR2_ceil_dim.csv'])
        # specifiy the sequence of gestures to be seen in the videos
        self.testLabelsKubik = np.array([[4,3,2,1],[4,3,2,1], [4,3,2,1], [4,3,2,1]])
        
        
        # The training files from Venzke
        self.trainingVenzke = np.array(['data/trainingVenzke/Compound_Garbage_25cm_190117_Annotated.csv',
                                        'data/trainingVenzke/Compound_Garbage_181128_Annotated.csv',
                                        'data/trainingVenzke/Compound_LRRL_Arm_25cm_Annotated_190117.csv',
                                        'data/trainingVenzke/Compound_LRRL_Finger_3cm_Annotated_181128.csv',
                                        'data/trainingVenzke/Compound_LRRL_Hand_5cm_Annotated_181128.csv',
                                        'data/trainingVenzke/Compound_TBBT_Arm_25cm_190117_Annotated.csv',
                                        'data/trainingVenzke/Compound_TBBT_Finger_3cm_Annotated_181128.csv',
                                        'data/trainingVenzke/Compound_TBBT_Hand_5cm_Annotated_181128.csv'
                                        ])
        


        # comment on eva data: Eva recorded with a 9 pixel and with a 16 pixel camera gestures. From the 16 pixel camera she just took 9 pixels so they can be used here as well    
        # Also she did every gesture with a finger and with a hand       
        self.evaHand = np.array(['data/dataEva9pixel/LRRL_finger_3cm_highBrightness_fast-annotated.csv',  # use for test data
                             'data/dataEva9pixel/LRRL_finger_3cm_highBrightness-annotated.csv',
                             'data/dataEva9pixel/LRRL_finger_5cm_highBrightness_fast-annotated.csv',
                             'data/dataEva9pixel/LRRL_finger_5cm_highBrightness-annotated.csv',
                             'data/dataEva9pixel/LRRL_finger_10cm_highBrightness-annotated.csv',
                             'data/dataEva9pixel/LRRL_finger_10cm_highBrightness-annotated.csv',
                             'data/dataEva9pixel/LRRL_finger_20cm_highBrightness-annotated.csv',
                             'data/dataEva9pixel/TBBT_finger_3cm_highBrightness-annotated.csv',
                             'data/dataEva9pixel/TBBT_finger_5cm_highBrightness_fast-annotated.csv',
                             'data/dataEva9pixel/TBBT_finger_10cm_highBrightness-annotated.csv',
                             'data/dataEva9pixel/TBBT_finger_20cm_highBrightness-annotated.csv',
                             'data/dataEva9pixel/TBBT_finger_3cm_highBrightness_fast-annotated.csv',
                             'data/dataEva9pixel/TBBT_finger_5cm_highBrightness-annotated.csv',
                             'data/dataEva9pixel/TBBT_finger_10cm_highBrightness_fast-annotated.csv',
                             'data/dataEva9pixel/TBBT_finger_20cm_highBrightness_fast-annotated.csv',                            
                             ])

        self.evaFinger = np.array(['data/dataEva9pixel/LRRL_hand_5cm_highBrightness_white-annotated.csv',
                                 'data/dataEva9pixel/LRRL_hand_5cm_highBrightness-annotated.csv',
                                 'data/dataEva9pixel/LRRL_hand_5cm_lowBrightness-annotated.csv',
                                 'data/dataEva9pixel/LRRL_hand_10cm_highBrightness_fast-annotated.csv',
                                 'data/dataEva9pixel/LRRL_hand_10cm_highBrightness_white-annotated.csv',
                                 'data/dataEva9pixel/LRRL_hand_10cm_highBrightness-annotated.csv',
                                 'data/dataEva9pixel/LRRL_hand_10cm_lowBrightness-annotated.csv',
                                 'data/dataEva9pixel/LRRL_hand_20cm_highBrightness_fast-annotated.csv',
                                 'data/dataEva9pixel/LRRL_hand_20cm_highBrightness_white-annotated.csv',
                                 'data/dataEva9pixel/LRRL_hand_20cm_highBrightness-annotated.csv',
                                 'data/dataEva9pixel/LRRL_hand_20cm_lowBrightness-annotated.csv',
                                 'data/dataEva9pixel/LRRL_hand_30cm_highBrightness_fast-annotated.csv',
                                 'data/dataEva9pixel/LRRL_hand_30cm_highBrightness-annotated.csv',
                                 'data/dataEva9pixel/LRRL_hand_30cm_lowBrightness-annotated.csv',                                 
                                 'data/dataEva9pixel/TBBT_hand_3cm_highBrightness_fast-annotated.csv',
                                 'data/dataEva9pixel/TBBT_hand_3cm_highBrightness-annotated.csv',
                                 'data/dataEva9pixel/TBBT_hand_3cm_lowBrightness-annotated.csv',
                                 'data/dataEva9pixel/TBBT_hand_5cm_highBrightness_fast-annotated.csv',
                                 'data/dataEva9pixel/TBBT_hand_5cm_highBrightness-annotated.csv',
                                 'data/dataEva9pixel/TBBT_hand_5cm_lowBrightness-annotated.csv',
                                 'data/dataEva9pixel/TBBT_hand_10cm_highBrightness_fast.csv',
                                 'data/dataEva9pixel/TBBT_hand_10cm_highBrightness-annotated.csv',
                                 'data/dataEva9pixel/TBBT_hand_10cm_lowBrightness-annotated.csv',
                                 'data/dataEva9pixel/TBBT_hand_20cm_highBrightness_white-annotated.csv',
                                 'data/dataEva9pixel/TBBT_hand_20cm_highBrightness-annotated.csv',
                                 'data/dataEva9pixel/TBBT_hand_20cm_lowBrightness-annotated.csv',
                                 'data/dataEva9pixel/TBBT_hand_30cm_highBrightness_white-annotated.csv',
                                 'data/dataEva9pixel/TBBT_hand_30cm_lowBrightness-annotated.csv',
                                 'data/dataEva9pixel/LRRL_hand_40cm_highBrightness-annotated.csv',
                                 'data/dataEva9pixel/LRRL_hand_40cm_lowBrightness-annotated.csv',
                                 'data/dataEva9pixel/TBBT_hand_40cm_highBrightness_fast-annotated.csv',
                                 'data/dataEva9pixel/TBBT_hand_40cm_highBrightness_white-annotated.csv',
                                 'data/dataEva9pixel/TBBT_hand_5cm_highBrightness_white-annotated.csv',
                                 'data/dataEva9pixel/TBBT_hand_10cm_highBrightness_white-annotated.csv',
                                 'data/dataEva9pixel/TBBT_hand_20cm_highBrightness_fast-annotated.csv',
                                 'data/dataEva9pixel/TBBT_hand_30cm_highBrightness_fast-annotated.csv',
                                 'data/dataEva9pixel/TBBT_hand_30cm_highBrightness-annotated.csv',
                                 'data/dataEva9pixel/LRRL_hand_40cm_highBrightness_fast-annotated.csv',
                                 'data/dataEva9pixel/LRRL_hand_50cm_lowBrightness-annotated.csv',
                                 'data/dataEva9pixel/TBBT_hand_40cm_lowBrightness-annotated.csv',
                                 'data/dataEva9pixel/LRRL_hand_3cm_highBrightness_fast-annotated.csv',
                                 'data/dataEva9pixel/LRRL_hand_3cm_highBrightness-annotated.csv',
                                 'data/dataEva9pixel/LRRL_hand_3cm_lowBrightness-annotated.csv',
                                 'data/dataEva9pixel/LRRL_hand_5cm_highBrightness_fast-annotated.csv'])
                
        # of the recordings for the 16 pixel camera, some were garbage scenes. However, they were not included to avoid giving the camera a bias towards classifying everything as garbage
        self.evaGarbage = np.array(['data/dataEva16pixel/Garbage_lt3cm_halfBrightness_0.csv',
                                    'data/dataEva16pixel/Garbage_lt3cm_halfBrightness_1.csv',
                                    'data/dataEva16pixel/Garbage_lt3cm_halfBrightness_2.csv',
                                    'data/dataEva16pixel/Garbage_lt3cm_halfBrightness_3.csv',
                                    'data/dataEva16pixel/Garbage_lt5cm_halfBrightness_0.csv',
                                    'data/dataEva16pixel/Garbage_lt5cm_halfBrightness_1.csv',
                                    'data/dataEva16pixel/Garbage_lt5cm_halfBrightness_2.csv',
                                    'data/dataEva16pixel/Garbage_lt5cm_halfBrightness_3.csv'])
        self.evaGarbageLabel = np.array([0]) # only garbage
        
        
        
        # arrays where the cleaned data shall be saved (gestures only, 20 frames per gesture)
        self.trainingX = np.array([])
        self.trainingY = np.array([])
        self.testX = np.array([])
        self.testY = np.array([])


    """
    readGestureFile
    Read a file and read all the lines
    Feed each frame into the buffer
    Whenever there is an event happening (buffer says when), add the fitting label 
    -> the sequence specifies, in which order those events
    happen (more events can happen than the length of the sequence, it repeats)

    -> Some files (for example from Venzke) are annotated and need no sequence so they have to be treated differently by this function
    -> line[9] returns which gesture is happening right now.


    But buffer only says a gesture has happend three frames after the gesture has happend
    So we need to keep track of line[9] FIVE rows ago! (that explains columnTen[rowCounter-5])
    @param author: string who created the file -> differences in processing it
    @param labels: for the files that are not annotated, the sequence provides the information which gesture was performed, else it is just an empty array
    @param generate_data: If yes: create synthetic data by roating and mirroring
    """
    def readGestureFile(self,path, labels,author,generate_data):
        # create the buffer
        buff = frameBuffer.CombindedFrameBuffer(0.01, 0.01, 0.1)
        # read file and get csv reader
        file = open(path)
        cr = csv.reader(file, delimiter=',')

        # the gestures and labels will be saved here
        gesturesLST = np.array([])
        labelsLST = np.array([])

        # keep track of where we are and at which gesture (if labels are specified) we are
        counter = 0
        rowCounter = 0

        # for annotated files we need to store the 10th column to keep track of the gestures
        columnTen = np.array([])

        for line in cr:
            completeLine = line
            try:
                line = line[0:9]            
            except:
                # end of file reached (some might be faulty in the last line), so we can just return what we have
                return gesturesLST,labelsLST
    
            if author == "venzke" or author == "eva":
                try:
                    columnTen = np.append(columnTen,completeLine[9])
                except:
                    break
            # count in which row we are
            rowCounter += 1

            # feed frame into buffer
            frame = PixelFrame(line)

            # check if buffer detected gesture
            if buff.feedFrame(frame):
                # true if an event is FINISHED AND at least 3 frames where nothing happend were processed
                # get those and add a label
                ges =buff.get_fsBuffer(20) # get 20 frame
                
                # check how we have to add the right label
                if author == "kubik" or author == "klisch":
                    label = labels[counter%len(labels)]
                elif author == "venzke" or author == "eva":
                        label = int(float(columnTen[rowCounter-5]))
                        if label == 9 or label == '9':
                        # convert to "no gesture"
                            label = 0
                if not generate_data:
                    labelsLST = np.append(labelsLST,label)     
                    gesturesLST = np.append(gesturesLST,ges)

                
                elif generate_data:    
                    # DISCLAIMER: I DID NOT USE AND I DID NOT TEST THE GENERATE DATA PROCEDURE. 
                    # rotate the szene 3 times
                    for rot in range(0,4):
                        # get the original scene before each call of rotate Gesture
                        gesture = np.copy(ges)
                        rotated = buff.rotateGesture(gesture, rot=rot)
                        gesturesLST = np.append(gesturesLST,rotated)
                        rotatedLabel = frameBuffer.rotate_label(label, rot)
                        labelsLST = np.append(labelsLST,rotatedLabel)
                        
                        # every rotated szene can be mirrored once horizontally and vertically                        
                        mirrored = frameBuffer.mirror_szene(rotated,"horizontal")
                        mirroredLabel = frameBuffer.mirror_label(rotatedLabel,"horizontal")
                        gesturesLST = np.append(gesturesLST,mirrored)
                        labelsLST = np.append(labelsLST,mirroredLabel)

                        mirrored = frameBuffer.mirror_szene(rotated,"vertical")
                        mirroredLabel = frameBuffer.mirror_label(rotatedLabel,"vertical")
                        gesturesLST = np.append(gesturesLST,mirrored)
                        labelsLST = np.append(labelsLST,mirroredLabel)

                counter = counter + 1
        return gesturesLST,labelsLST

   
    """
    collectAllGestures
    Function that employs other functions of this class to gather all the 
    test and train data and return it
    Here one can actually decide on which data one wants to use
    """
    def collectAllGestures(self):        

        # collect TRAINING DATA Kubik
        for f,labels in zip(self.trainingFilesKubik,self.trainingLabelsKubik):
            gestures,label = self.readGestureFile(f,labels,"kubik",False)
            self.trainingX = np.append(self.trainingX,gestures)
            self.trainingY = np.append(self.trainingY,label)
        self.trainingX = np.reshape(self.trainingX,(-1,180))
        

        
        # collect MORE DATA Kubik
        for f,labels in zip(self.testKubik,self.testLabelsKubik):
            gestures,label = self.readGestureFile(f,labels,"kubik",False)
            self.trainingX = np.append(self.trainingX,gestures)
            self.trainingY = np.append(self.trainingY,label)
        self.trainingX = np.reshape(self.trainingX,(-1,180))

        # collect test kubik and klisch used 
        for f in self.testKlisch:
            gestures,label = self.readGestureFile(f,np.array([]),"venzke",False)
            self.testX = np.append(self.testX,gestures)
            self.testY = np.append(self.testY,label)
        self.testX = np.reshape(self.testX,(-1,180))


        # collect Venzke data
        for f in self.trainingVenzke:
            gestures,labels = self.readGestureFile(f,np.array([]),"venzke",False)
            self.trainingX = np.append(self.trainingX,gestures)
            self.trainingY = np.append(self.trainingY,labels)
        self.trainingX = np.reshape(self.trainingX,(-1,180))
        
        
        # collect Eva data
        for f in self.evaHand:
            gestures,labels = self.readGestureFile(f,np.array([]),"eva",False)
            self.trainingX = np.append(self.trainingX,gestures)
            self.trainingY = np.append(self.trainingY,labels)
        self.trainingX = np.reshape(self.trainingX,(-1,180))

        # collect Eva data
        for f in self.evaFinger:
            gestures,labels = self.readGestureFile(f,np.array([]),"eva",False)
            self.trainingX = np.append(self.trainingX,gestures)
            self.trainingY = np.append(self.trainingY,labels)
        self.trainingX = np.reshape(self.trainingX,(-1,180))

        # collect Eva garbage data -> and rotate the garbage because she only recorded it from one corner
        """for f in self.evaGarbage:
            gestures,labels = self.readGestureFile(f,self.evaGarbageLabel,"kubik",True) # treat it as kubik file, no annotations in file
            self.trainingX = np.append(self.trainingX,gestures)
            self.trainingY = np.append(self.trainingY,labels)
        self.trainingX = np.reshape(self.trainingX,(-1,180))"""
        

        return self.trainingX,self.trainingY,self.testX,self.testY
    

    """
    printAllData
    Function I created for debug purpose. 
    It is capable of writing all the testX and trainingX data in csv files
    Furthermore it adds which sequence is to be seen. The video player can play
    the videos then back and we can witnes if the data is any good
    And if the relation between gesture and trainY fits
    """
    def printAllData(self):
        f = open('gatherGesturesOutput/videoTestData.csv','w')
        writer = csv.writer(f)
        for r in range(len(self.testX)):
            for index in range(0,180,9):
                #print((self.testX[r][index:index+9]*1024).astype(int))
                #print(self.testY[r])
                frame = (self.testX[r][index:index+9]*1024).astype(int)
                ges = int(float(self.testY[r]))
                row = np.append(frame,ges)
                #    print("except")
                writer.writerow(row)
        f = open('gatherGesturesOutput/videoTrainData.csv','w')
        writer = csv.writer(f)
        for r in range(len(self.trainingX)):
            for index in range(0,180,9):
                # add the fitting label to each gesture, denormalize and convert to int
                frame = (self.trainingX[r][index:index+9]*1024).astype(int) #*1024
                ges = int(float(self.trainingY[r]))
                row = np.append(frame,ges)                
                writer.writerow(row)

    """
    printTestArduino
    This function has the purupose to write the test data to an arduino .h file
    It writes test.h and test2.h and test3.h because it is too big to get processed by the arduino at once
    """
    def printTestArduino(self):
        #train = self.trainingX[0:32]
        #trainy = self.trainingY[0:32]
        train = self.trainingX[32:64]
        trainy = self.trainingY[32:64]
        f = open("gatherGesturesOutput/train.h","w")
        writer = csv.writer(f,delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_NONE,escapechar=' ',lineterminator='\n')
        writer.writerow(["#ifndef TRAIN_H"])
        writer.writerow(["#define TRAIN_H"])
        writer.writerow(["#include <avr/pgmspace.h>"])
        end = int(len(train))
        writer.writerow(["const int numberOfTests = "+str(end)+";"])
        writer.writerow(["const int testset[] PROGMEM = {"])
        for r in range(end):
            for index in range(0,180,9):
                frame = (train[r][index:index+9]*1024).astype(int)
                string = ""
                for f in frame:
                    string +=str(f)+","
                if r == end-1 and index == 171:
                    writer.writerow([string[0:len(string)-1]])
                else:
                    writer.writerow([string])
        writer.writerow(["};"])
        string = ""
        for r in range(end):
            label = trainy[r]
            string+=str(label)+","
        writer.writerow(["const unsigned char testlabel[] PROGMEM = {"+str(string[0:len(string)-1]+"};")])
        writer.writerow(["#endif"])

        f = open("gatherGesturesOutput/test2.h","w")
        writer = csv.writer(f,delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_NONE,escapechar=' ',lineterminator='\n')
        writer.writerow(["#ifndef TEST_H"])
        writer.writerow(["#define TEST_H"])
        writer.writerow(["#include <avr/pgmspace.h>"])
        start = int(len(train)/3)
        end = int(2*len(train)/3)
        writer.writerow(["const int numberOfTests = "+str(end-start)+";"])
        writer.writerow(["const int testset[] PROGMEM = {"])
        for r in range(start,end):
            for index in range(0,180,9):
                frame = (train[r][index:index+9]*1024).astype(int)
                string = ""
                for f in frame:
                    string +=str(f)+","
                if r == len(train)-1 and index == 171:
                    writer.writerow([string[0:len(string)-1]])
                else:
                    writer.writerow([string])
        writer.writerow(["};"])
        string = ""
        for r in range(start,end):
            label = self.testY[r]
            string+=str(label)+","
        writer.writerow(["const unsigned char testlabel[] PROGMEM = {"+str(string[0:len(string)-1]+"};")])
        writer.writerow(["#endif"])


        
        f = open("gatherGesturesOutput/test.h","w")
        writer = csv.writer(f,delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_NONE,escapechar=' ',lineterminator='\n')
        writer.writerow(["#ifndef TEST_H"])
        writer.writerow(["#define TEST_H"])
        writer.writerow(["#include <avr/pgmspace.h>"])
        end = int(len(self.testX)/3)
        writer.writerow(["const int numberOfTests = "+str(end)+";"])
        writer.writerow(["const int testset[] PROGMEM = {"])
        for r in range(end):
            for index in range(0,180,9):
                frame = (self.testX[r][index:index+9]*1024).astype(int)
                string = ""
                for f in frame:
                    string +=str(f)+","
                if r == end-1 and index == 171:
                    writer.writerow([string[0:len(string)-1]])
                else:
                    writer.writerow([string])
        writer.writerow(["};"])
        string = ""
        for r in range(end):
            label = self.testY[r]
            string+=str(label)+","
        writer.writerow(["const unsigned char testlabel[] PROGMEM = {"+str(string[0:len(string)-1]+"};")])
        writer.writerow(["#endif"])
        
        f = open("gatherGesturesOutput/test2.h","w")
        writer = csv.writer(f,delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_NONE,escapechar=' ',lineterminator='\n')
        writer.writerow(["#ifndef TEST_H"])
        writer.writerow(["#define TEST_H"])
        writer.writerow(["#include <avr/pgmspace.h>"])
        start = int(len(self.testX)/3)
        end = int(2*len(self.testX)/3)
        writer.writerow(["const int numberOfTests = "+str(end-start)+";"])
        writer.writerow(["const int testset[] PROGMEM = {"])
        for r in range(start,end):
            for index in range(0,180,9):
                frame = (self.testX[r][index:index+9]*1024).astype(int)
                string = ""
                for f in frame:
                    string +=str(f)+","
                if r == len(self.testX)-1 and index == 171:
                    writer.writerow([string[0:len(string)-1]])
                else:
                    writer.writerow([string])
        writer.writerow(["};"])
        string = ""
        for r in range(start,end):
            label = self.testY[r]
            string+=str(label)+","
        writer.writerow(["const unsigned char testlabel[] PROGMEM = {"+str(string[0:len(string)-1]+"};")])
        writer.writerow(["#endif"])

        f = open("gatherGesturesOutput/test3.h","w")
        writer = csv.writer(f,delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_NONE,escapechar=' ',lineterminator='\n')
        writer.writerow(["#ifndef TEST_H"])
        writer.writerow(["#define TEST_H"])
        writer.writerow(["#include <avr/pgmspace.h>"])
        start = int(2*len(self.testX)/3)
        end = int(len(self.testX))
        writer.writerow(["const int numberOfTests = "+str(end-start)+";"])
        writer.writerow(["const int testset[] PROGMEM = {"])
        for r in range(start,end):
            for index in range(0,180,9):
                frame = (self.testX[r][index:index+9]*1024).astype(int)
                string = ""
                for f in frame:
                    string +=str(f)+","
                if r == len(self.testX)-1 and index == 171:
                    writer.writerow([string[0:len(string)-1]])
                else:
                    writer.writerow([string])
        writer.writerow(["};"])
        string = ""
        for r in range(start,end):
            label = self.testY[r]
            string+=str(label)+","
        writer.writerow(["const unsigned char testlabel[] PROGMEM = {"+str(string[0:len(string)-1]+"};")])
        writer.writerow(["#endif"])


    """
    saveAsArray
    The synthetic_annotated.csv file is too big to read it everytime it is needed
    So we read it here once and save it as array
    """
    def saveAsArray(self):
        gestures,labels = self.readGestureFile("data/trainingVenzke/synthetic_annotated.csv",np.array([]),"venzke",False)
        gestures = np.reshape(gestures,(-1,180))
        
        np.save("gatherGesturesOutput/synthetic_annotatedX.npy",gestures)
        np.save("gatherGesturesOutput/synthetic_annotatedY.npy",labels)
        print("finished")
    
    """
    loadsynthetic array
    It is much simpler to just load an npy file instead of processing the synthetic csv file every time it is needed
    This function loads the array and returns it
    """
    def loadsyntheticArray(self):
        path = "gatherGesturesOutput/synthetic_annotatedX.npy"
        with open(path, 'rb') as f:
            x = np.load(f)
        path = "gatherGesturesOutput/synthetic_annotatedY.npy"
        with open(path, 'rb') as f:
            y = np.load(f)
        return x,y


# debug
if __name__ == "__main__":
    g = gatherGestures()
    g.collectAllGestures()
    g.printAllData()
    g.printTestArduino()
    #g.saveAsArray()
    #g.loadsyntheticArray()


