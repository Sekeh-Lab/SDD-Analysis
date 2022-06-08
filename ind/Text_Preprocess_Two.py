#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 03:37:59 2021

@author: josh
"""


import re
import numpy as np
from csv import reader
import pandas as pd
import os





def preprocess(sceneName, inputFile):
   

    """
    Section 2: Extract every 10th frame from the previous output file to emulate 2.5 frames per second
    """
        
    fileI = open((sceneName + inputFile), 'r')
    fileO = open((sceneName + 'pos_data_temp.csv'), 'w')
    
    lines = fileI.readlines()
    
    valid = re.compile(r"^([0-9]+),([0-9]+),(\S+),(\S+),(\S+)\s+")
    
    for line in lines:
        if(valid.match(line)):
            matchText = valid.match(line)

            frame = matchText.group(1)
            
            if(int(frame) % 10 == 0):
                fileO.writelines(line)
    
    fileI.close()
    fileO.close()
    return 0





i=0


for root, subdir, file in os.walk("./"):
    if len(file)>0:
        print(i)
        i+=1

        print(root)
        print(subdir)
        print(file[0])
        for f in file:
            if f == "pos_data_interp.csv":
                preprocess((root+"/"), f)
  

