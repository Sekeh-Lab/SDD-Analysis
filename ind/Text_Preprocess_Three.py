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
    Section 3: Transpose the resulting data
    """
         
    csv = pd.read_csv((sceneName + inputFile), header = None, delimiter = r',')

    csv = csv.transpose()
    

    csv.to_csv((sceneName + "pos_data.csv"), header = False, index = False)
                
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
            if f == "pos_data_temp.csv":
                preprocess((root+"/"), f)
  