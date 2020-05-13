#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 
import os 
from pathlib import Path


# In[2]:


cdir = str(os.getcwd())

load_name = '/Step1'

flp = cdir + load_name
#print(fln)

file_names = sorted(os.listdir(flp)) 

# remove any non-JPG files: 
file_names = [f for f in file_names if '.jpg' in f] 

for i in range(len(file_names)):
    print(file_names[i])
    fln = flp + '/' + file_names[i]
    
    img = cv2.imread(fln)
    height, width, _ = n=img.shape
    
    if height > width:
        img_rotate_90 = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(fln, img_rotate_90)


# In[ ]:




