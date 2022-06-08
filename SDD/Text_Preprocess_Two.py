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





def preprocess_raw(sceneName, inputFile):
   

    """
    Section 2: Extract every 12th frame from the previous output file to emulate the proper FPS
    """
        
    fileI = open((sceneName + 'pos_data_interp.csv'), 'r')
    fileO = open((sceneName + 'pos_data_temp.csv'), 'w')
    
    lines = fileI.readlines()
    
    valid = re.compile(r"^([0-9]+),([0-9]+),(\S+),(\S+),(\S+)\s+")

    for line in lines:
        if(valid.match(line)):
            matchText = valid.match(line)

            frame = matchText.group(1)
            
            if(int(frame) % 12 == 0):
                fileO.writelines((matchText.group(1) + ' ' + matchText.group(2) + ' ' + matchText.group(3) + ' ' + matchText.group(4) + ' ' + matchText.group(5) +'\n'))
                
    return 0

i=0

for root, subdir, file in os.walk("./data"):
  if len(file)>0:
      print(i)
      i+=1

      print(root)
      print(subdir)
      print(file[0])
      for f in file:
          if f == "pos_data_interp.csv":
              preprocess_raw((root+"/"), f)
  