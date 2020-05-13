#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2 
import os 
from pathlib import Path
import shutil
import numpy as np


# In[2]:


cdir = str(os.getcwd())

flp = cdir + '/Bee-frames-yz'

file_names = sorted(os.listdir(flp)) 

# remove any non-JPG files: 
file_names = [f for f in file_names if '.jpg' in f] 

np.random.seed(148)
np.random.shuffle(file_names)

selected_names = file_names[300:400]

for file in selected_names:
    
    shutil.move(flp+'/'+file, cdir+'/'+'Step1/')

