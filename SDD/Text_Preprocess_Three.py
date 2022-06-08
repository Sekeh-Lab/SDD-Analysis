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
    Section 3: Transpose the resulting data and remove all trajectory data after any mid-trajectory "lost" coordinates
    """
    colnames = ['frame', 'id', 'x', 'y', 'class']
    csv = pd.read_csv((sceneName + "pos_data_temp.csv"), header = None, names = colnames, delimiter = r' ')

    """
    For each agent ID, iterate over the annotations. If lost coordinates occur in the middle of a trajectory, they got filtered out in
       the first processing script. In such a case, the gap in frame # between the coordinates before and after the lost portion will be 
        greater than 12 (since we took every 12th frame, and filtered out some frames in that portion). We mask out the remainder of the trajectory
        after this has been detected. This won't remove any coordinates unless used on the corrected version of the data, since otherwise
        every subsequent coordinate in pos_data_temp.csv would be exactly 12 frames apart, so no masking will occur.
    """
    obspred = []
    for i in csv['id'].unique():
        rem = 0
        temp = csv[csv['id']==i].copy()
        for j in range(1, len(temp)):
            if (temp.iloc[j,0] - temp.iloc[j-1,0]) > 12:
               rem = 1 
               ### Sets a placeholder value to indicate frames to be masked.
             if rem == 1:
               temp.iloc[j,0] = -100
        temp = temp[temp['frame'] != -100]
        if len(temp.values.tolist()) >= 20:
            obspred.extend(temp.iloc[:20].values)
        else:
            print("too short for: ", i)
            
    ### Saves the output with the class labels        
    obspred_df = pd.DataFrame(obspred) 
    obspred_df.to_csv((sceneName + "data_full.csv"), sep=' ', header = False, index = False)
    
    ### Saves the output without the class labels
    no_class = obspred_df.iloc[:,:4]
    no_class.to_csv((sceneName + "data.csv"), sep=' ', header = False, index = False)
    return 0

i=0

for root, subdir, file in os.walk("./data/"):
  if len(file)>0:
      print(i)
      i+=1

      print(root)
      print(subdir)
      print(file[0])
      for f in file:
          if f == "pos_data_temp.csv":
              preprocess_raw((root+"/"), f)
  