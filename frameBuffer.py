""""
Comment by Anton:
This file provides multiple event recognizing buffers of which only the 'combinedFrameBuffer' was really used
The others were mostly proof of concept and remained stubs


Author: Kubik, Anton Giese changed some things for example shape of gestures
"""

from collections import deque
import numpy as np
import math
import csv

class EMAFrameBuffer():
    def __init__(self, alpha_ema, alpha_sd, delta_pix, queue_len):
        self.buffer = deque([], maxlen=20)
        self.alpha_ema = alpha_ema
        self.alpha_sd = alpha_sd
        self.delta_pix  = delta_pix

        self.ema = 0
        self.sd = 0

        self.event = False

    def feedFrame(self, frame):
        self.buffer.append(frame)
        self.ema = self.ema * (1-self.alpha_ema) + frame.mean * self.alpha_ema
        self.sd = self.sd * (1-self.alpha_sd) + abs(frame.mean - self.ema) * self.alpha_sd

        if len(self.buffer) == 20:

            if frame.mean < self.ema - self.sd * 1.5:
               self.event = True

            elif self.event and not frame.pixeldiff(self.buffer[-2], self.delta_pix):
                self.event = False
                return True

        return False

    def flattenBuffer(self):
        rd = np.zeros((1, 180))
        i = 0
        while i < 20:
            j = 0
            while j < 9:
                rd[0][i * 9 + j] = self.buffer[i].data[j]
                j = j + 1
            i = i + 1
        return rd

class continousEvalBuffer():
    def __init__(self):
        self.q = deque([], maxlen=20)
        
    def feedFrame(self, frame):
        self.q.append(frame)
        if len(self.q) == 20:
            return True
        else:
            return False

    def get_buffer(self):
        ret = np.empty((1,180))
        i = 0
        j = 0
        while i < 20:
            while j < 9:
                ret[0][j+i*9] = self.q[i].data[j]
                j = j + 1
            i = i + 1
        return ret

class CombindedFrameBuffer():
    def __init__(self, alpha, delta, margin):
        self.treshlo = 0
        self.treshhi = 0
        self.ema = 0
        self.alpha = alpha
        self.delta = delta
        self.margin = margin
        self.buff = deque([])
        self.lastframe = None
        self.gestureflag = False
        self.recCountdown = 0
        self.frameCounter = 0

    """
    feedFrame
    When there is a new frame available you can feed it into the buffer instance
    The frame is an object of the class Pixelframe

    If its the first frame we need to initialize some variables

    If no event is happening (when the mean of the current frame is between high and low threshold)
    -> If an event was happening UNTIL this frame (this frame is first frame of no event anymore),
    set cooldown time of three frames, reset flag
    -> If no event was happening until this frame: check if new frame differs from last one, if yes
    adjust thresholds

    If it detects an event (mean is higher than high threshold or lower than low threshold)
    -> set flag

    AFTER the cooldown: return true, telling the calling instance that an event just happend,
    empty the buffer so that only 3 pictures are left

    Default: empty buffer so that only three frames are left (except for if an event is happening of course)

    """
    def feedFrame(self, frame):
        self.frameCounter += 1
        self.buff.append(frame)
        if self.lastframe is None:  
            # if this is the first frame to be fed: initialize a few things
            self.lastframe = frame
            self.ema = frame.mean
            self.treshlo = self.ema * (1 - self.margin)
            self.treshhi = self.ema * (1 + self.margin)

        elif frame.mean >= self.treshlo and frame.mean <= self.treshhi:  
            # if there isn't an event happening right now...
            if self.gestureflag:  
                # but there has been right the frame before
                # set "cooldown" of three frames -> tell calling instance an event happend in three frames
                #print("At frame {} the event is finished".format(self.frameCounter))
                self.recCountdown = 3  # fixme 5

                # reset flag
                self.gestureflag = False

            if self.lastframe.pixeldiff(frame, self.delta):  # and the image is still Anton: ???? what does he mean
                # it variates from frame before, set new tresholds and new mean
                # the mean is a percentage of the old mean
                #print("At frame {} pixeldiff trigger".format(self.frameCounter))
                self.ema = self.ema * (1-self.alpha) + frame.mean * self.alpha
                self.treshlo = self.ema * (1 - self.margin)
                self.treshhi = self.ema * (1 + self.margin)
        
        elif (frame.mean < self.treshlo or frame.mean > self.treshhi) and not self.gestureflag:
            # start of an event was detected
            #print("At frame {} the start of an event was detected".format(self.frameCounter))
            self.gestureflag = True
            #self.buff = self.buff.maxlen = 5

        if(self.recCountdown > 0):
            self.recCountdown = self.recCountdown -1
            if self.recCountdown == 0:
                # cooldown countdown is OVER
                #self.buff = deque([]) # todo delete (dbg)
                #print("At frame {} buffer returns true".format(self.frameCounter))

                #tell calling instance something happened
                #self.lastframe = frame # doesnt change anything in the result
                return True
            
        # empty buffer only if no cooldown and no event right now
        while not self.gestureflag and self.recCountdown == 0 and len(self.buff) > 3:  # fixme 5
            # empty buffer (only 3 frames left in buffer)
            self.buff.popleft()
            #print("delete element at frame",self.frameCounter)

        #print("länge des buff: ",len(self.buff))
        if len(self.buff) > 100:
            print('WARN: buffer got too big! resetting')
            self.lastframe = frame
            self.ema = frame.mean
            self.treshlo = self.ema * (1 - self.margin)
            self.treshhi = self.ema * (1 + self.margin)
            self.buff = deque([])
        #self.lastframe = frame
        return False

    """
    get_buffer
    Purpose: copy the the whole buffer
    """
    def get_buffer(self):
        rd = np.zeros((len(self.buff), 9))
        i = 0
        while i < len(self.buff):
            j = 0
            while j < 9:
                # copy single pixel values
                rd[i][j] = self.buff[i].data[j]
                j = j + 1
            i = i + 1
        return rd

    """
    get_scaled_buffer
    The neural network wants 20 input frames. Since in real life events might be longer or shorter
    we need to artificially interpolate the existing frames to 20 
    This can be done with pseudoindexes
    Those can be calcualted using the linear function
    0 = m*0+b
    n-1 = m*20+b
    where n is the number of actual frames we recorded
    """
    def get_scaled_buffer(self, n, rot):
        rotind = [6,3,0,7,4,1,8,5,2]
        pseudoindexes = list() 
        b = self.get_buffer()
        #print("Buffershape: ",b.shape)
        
        # calculate the pseudoindexes
        #for i in range(1,n+1):
        for i in range(0,n):
            # calculate the pseudo indexes -> the indexes the frames 0..n WOULD have if the nth frame would
            # be the highest frame in the buffer
            #pseudoindex = (b.shape[0]-1)/(n-1)*i+((20-b.shape[0])/19) (with different assumption)
            pseudoindex = i*(b.shape[0]-1)/(n-1)
            pseudoindexes.append(pseudoindex)
        
        # due to inaccuracy the last entry might glitch, so we manually set it on the last frame NO RIGHT NOW
        #pseudoindexes[len(pseudoindexes)-1] = b.shape[0] (with different assumption)

        #print("Pseudindexes: ",pseudoindexes)
        #print("länge des gesture arrays: ",len(pseudoindexes))

        # calculate interpolated frames
        gesture = np.empty((n,9))
        i = 0
        while i < gesture.shape[0]: #20
            j = 0
            while j < gesture.shape[1]: #9
                if isinstance(pseudoindexes[i],int):
                    # if pseudoindex is an acutal int, we can just copy the according frame

                    #gesture[i][j] = b[int(pseudoindexes[i])-1][j] (with different assumption)
                    gesture[i][j] = b[int(pseudoindexes[i])][j]

                else:
                    # normally its not an int

                    d1 = pseudoindexes[i] % 1 # get only the digits BEHIND the comma, how close the
                                              # pseudoindex is to the actual index
                    d2 = 1-d1
                    #gesture[i][j] = b[int(math.floor(pseudoindexes[i]))-1][j] * d2 + b[int(math.ceil(pseudoindexes[i]))-1][j] * d1 (with different assumption)
                    gesture[i][j] = b[int(math.floor(pseudoindexes[i]))][j] * d2 + b[int(math.ceil(pseudoindexes[i]))][j] * d1
                                        
                j = j + 1
            i = i + 1
        # if desired: rotate the frame to create synthetic data
        while rot > 0:
            rotated_gesture = np.zeros((gesture.shape))
            for j in range(0,20):
                for i in range(0,9):
                    rotated_gesture[j][rotind[i]] = gesture[j][i]
                    print(gesture[j][i])
                gesture = rotated_gesture
            rot = rot - 1
        #gesture = np.reshape(gesture,(n,9))
        return gesture

    """
    Get buffer
    n: number of frames we want
    rot if we want the frames rotated or not
    """
    def get_fsBuffer(self, n, rot=0):
        #print(len(self.buff))
        #for pixelframe in self.buff:
            #print(pixelframe.data)
         #   d = pixelframe.data
          #  d = [a*1024 for a in d]
           # print(d)
           # d = list(map(int, d))
            #print(d)
            #self.writer.writerow(d)

        
        sb = self.get_scaled_buffer(n, rot)
        #print(sb*1024)
        #sb = sb*1024#[e*1024 for e in sb]
        #sb = sb.astype(int)
        #for row in sb:
           # self.writer.writerow(row)
            
        #print(sb)
        ret = np.ndarray.flatten(sb,'A')  #fixme debug only; remove!
        #print(ret)
        return ret
        #return self.get_buffer()

    def rotateGesture(self,frame,rot):
        if rot == 0:
            return frame
        gesture = np.reshape(np.copy(frame),(20,9))
        # if desired: rotate the frame to create synthetic data
        rotind = [6,3,0,7,4,1,8,5,2]
        rotated_gesture = np.ones((20,9))
        
        while rot > 0:
            for j in range(0,20):
                for i in range(0,9):
         #           print(gesture[j][i])
                    rotated_gesture[j][rotind[i]] = gesture[j][i]
            gesture = np.copy(rotated_gesture)
            rot = rot - 1
        #gesture = np.reshape(gesture,(n,9))
        #print(rotated_gesture)
        return rotated_gesture.flatten()


    def get_buffer_length(self):
        return len(self.buff)

    def clearBuffer(self): # todo cleanup
        self.ema = self.lastframe.mean
        self.treshlo = self.ema * (1 - self.margin)
        self.buff = deque([])
        return True

class SDFrameBuffer(CombindedFrameBuffer):
    def __init__(self, alpha, beta, delta, margin):
        def __init__(self, alpha, delta, margin):
            self.tresh = 0
            self.ema = 0
            self.sd = 1
            self.alpha = alpha
            self.delta = delta
            self.margin = margin
            self.buff = deque([])
            self.lastframe = None
            self.gestureflag = False
            self.recCountdown = 0

    def feedFrame(self, frame):
        return True

class fivePixelBuffer(CombindedFrameBuffer):
    def get_fsBuffer(self, n, rot=0):
        sb = self.get_scaled_buffer(n, rot)
        ret = np.empty(n*5)
        i = 0
        j = 0
        k = 0
        while i < n:
            while j < 8:
                if j!=0 and j!=2 and j!=6:
                    ret[k] = sb[i][j]
                    k = k +1
                j = j +1
            i = i+1
        return ret

def rotate_label(label, n):
    for i in range(0,n):
        if label == 0:
            label = 0
        elif label == 1:
            label = 4
        elif label == 2:
            label = 3
        elif label == 3:
            label = 1
        elif label == 4:
            label = 2
    return label

"""
mirror_szene
This function was implemented addiditonally by Anton 
The purpose is to mirror a szene
@param szene: the szene values
@param orientation: either horiontally or vertically
"""
def mirror_szene(szene,orientation):
    gesture = np.reshape(np.copy(szene),(20,9))
        # if desired: mirror the frame to create synthetic data
    mirrorIndexesHorizontal = [6,7,8,3,4,5,0,1,2]
    mirrorIndexesVertically = [2,1,0,5,4,3,8,7,6]
    mirrored_gesture = np.ones((20,9))    
    for j in range(0,20):
        # mirror each frame
        for i in range(0,9):
            # copy pixel to right place
            if orientation == "horizontal":
                mirrored_gesture[j][mirrorIndexesHorizontal[i]] = gesture[j][i]
            else:
                mirrored_gesture[j][mirrorIndexesVertically[i]] = gesture[j][i]
    return mirrored_gesture.flatten()

"""
mirror_label
The partner function to mirror_szene: take care of the label
"""
def mirror_label(label,orientation):
    if label == 0:
        return 0
    if orientation == "horizontal":
        if label == 4:
            return 3
        elif label == 3:
            return 4
        else:
            return label
    elif orientation == "vertical":
        if label == 1:
            return 2
        elif label == 2:
            return 1
        else:
            return label
