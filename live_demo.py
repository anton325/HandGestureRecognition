"""
live
Comment by anton:
this file is taken from Mr. Kubik
It initiates a serial connection and reads line after line of brightness values from the camera
Then it uses a loaded keras model to classify gestures

Author: Kubik, changes by Anton
"""


import serial
import serial.tools.list_ports

import keras
from Frame import PixelFrame
import numpy as np
import frameBuffer
from subprocess import Popen, PIPE
import time as t

presskeys = False

ports = serial.tools.list_ports.comports()

for port, desc, hwid in sorted(ports):
        print("{}: {} [{}]".format(port, desc, hwid))

# on windows
#ser = serial.Serial('/dev/ttyUSB0', 115200)
#ser = serial.Serial('/dev/ttyACM0', 115200)

# ON MAC: 
try:
    ser = serial.Serial('/dev/cu.usbmodem14201', 115200)
except:
    ser = serial.Serial('/dev/cu.usbmodem14101', 115200)





model = keras.models.load_model('quantizedModelFinal.h5')
model.compile(optimizer = 'adam',
                        loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                        metrics = ['accuracy'])
                        
#model = load_model('first_working_model.h5')


buffer = frameBuffer.CombindedFrameBuffer(0.01,0.01,0.1)


ser.flush()
ser.readline()  # um die erste abgeschnittene zeile loszuwerden
start = t.time() -1

while 1 == 1:
    end = t.time()
    #print("It took {}s for 1 frame which leads to a framerate of {}".format(end-start,1/(end-start)))
    start = t.time()
    line = ser.readline()
    #for p in line:
    #    print(type(p))
    
    #try:
    frame = PixelFrame(line)
    print("frame: ",line)
    #except Exception:
    #    print("io error")
    #    continue
        

    #try:
    if buffer.feedFrame(frame):
        print("gesture detected")        
        test = np.empty((1,180))
        test[0] = buffer.get_fsBuffer(20)
        if buffer.get_buffer_length() < 6:
            print('gesture too short - dropped')
        else:
            print(buffer.get_buffer_length())
            guess = model.predict_classes(test)
            print(model.predict_classes(test))
            if guess[0] == 3:
                print('oben -> unten')
                if presskeys:
                    p = Popen(['xte'], stdin=PIPE)
                    p.communicate("key Down\n")
            elif guess[0] == 4:
                print('unten -> oben')
                if presskeys:
                    p = Popen(['xte'], stdin=PIPE)
                    p.communicate("key Up\n")
            elif guess[0] == 1:
                print('links -> rechts')
                if presskeys:
                    p = Popen(['xte'], stdin=PIPE)
                    p.communicate("key Right\n")
            elif guess[0] == 2:
                print('rechts -> links')
                if presskeys:
                    p = Popen(['xte'], stdin=PIPE)
                    p.communicate("key Left\n")
            elif guess[0] == 0:
                print('event is not a gesture')
            print('\n')
        t.sleep(1.5)
    #except frameBuffer.FrameBufferException:
    #    print('gesture is taking too long - clearing buffer')
    #    buffer.clearBuffer()
    
