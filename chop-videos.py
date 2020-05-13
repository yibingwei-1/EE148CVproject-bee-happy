#!/usr/bin/env python
# coding: utf-8

import cv2 
import os 
from pathlib import Path


# In[2]:


cdir = str(os.getcwd())

load_name = ['/Bee-videos-yz','/Bee-videos-xz']
save_name = ['./Bee-frames-yz/','./Bee-frames-xz/']

for fi in range(2):

    flp = cdir + load_name[fi]
    #print(fln)
    
    file_names = sorted(os.listdir(flp)) 

    # remove any non-MOV files: 
    file_names = [f for f in file_names if '.MOV' in f] 

    for i in range(len(file_names)):
        print(file_names[i])
        fln = flp + '/' + file_names[i]
        cam = cv2.VideoCapture(fln)
        if cam.isOpened():
            print("Device Opened\n")
        else:
            print("Failed to open Device\n")

        # frame 
        currentframe = 0

        while(True): 

            # reading from frame 
            ret,frame = cam.read() 

            if ret: 
                # if video is still left continue creating images 
                name = save_name[fi] + str(i).zfill(3)+ str(currentframe).zfill(4) + '.jpg'
                #print ('Creating...' + name) 

                if currentframe%20 == 0:
                    # writing the extracted images 
                    cv2.imwrite(name, frame) 

                # increasing counter so that it will 
                # show how many frames are created 
                currentframe += 1
            else: 
                break

        cam.release() 
        cv2.destroyAllWindows()






