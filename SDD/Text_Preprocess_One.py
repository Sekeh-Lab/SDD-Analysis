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





def preprocess_raw(sceneName, inputFile, vers):

    
    
    if sceneName == "./annotations/coupa/video0/" or sceneName == "./annotations/coupa/video1/" or sceneName == "./annotations/coupa/video2/" or sceneName == "./annotations/coupa/video3/":
        range_x = 1980
        range_y = 1093
    
    elif sceneName == "./annotations/deathCircle/video0/":
        range_x = 1630
        range_y = 1948
        
    elif sceneName == "./annotations/deathCircle/video1/":
        range_x = 1409
        range_y = 1916
    
    elif sceneName == "./annotations/deathCircle/video2/":
        range_x = 1436
        range_y = 1959

    elif sceneName == "./annotations/deathCircle/video3/":
        range_x = 1400
        range_y = 1904
        
    elif sceneName == "./annotations/deathCircle/video4/":
        range_x = 1452
        range_y = 1994
        
    elif sceneName == "./annotations/gates/video0/" or sceneName == "./annotations/gates/video2/":
        range_x = 1325
        range_y = 1973
        
    elif sceneName == "./annotations/gates/video1/":
        range_x = 1425
        range_y = 1973
        
    elif sceneName == "./annotations/gates/video3/":
        range_x = 1432
        range_y = 2002
        
    elif sceneName == "./annotations/gates/video4/":
        range_x = 1434
        range_y = 1982
        
    elif sceneName == "./annotations/gates/video5/":
        range_x = 1426
        range_y = 2011
    
    elif sceneName == "./annotations/gates/video6/":
        range_x = 1326
        range_y = 2011
        
    elif sceneName == "./annotations/gates/video7/" or sceneName == "./annotations/gates/video8/":
        range_x = 1334
        range_y = 1982
    
    elif sceneName == "./annotations/hyang/video0/":
        range_x = 1455
        range_y = 1925
        
    elif sceneName == "./annotations/hyang/video1/":
        range_x = 1445
        range_y = 2002
        
    elif sceneName == "./annotations/hyang/video2/":
        range_x = 1433
        range_y = 841

    elif sceneName == "./annotations/hyang/video3/":
        range_x = 1433
        range_y = 741
        
    elif sceneName == "./annotations/hyang/video4/":
        range_x = 1340
        range_y = 1730
        
    elif sceneName == "./annotations/hyang/video5/":
        range_x = 1454
        range_y = 1991
        
    elif sceneName == "./annotations/hyang/video6/":
        range_x = 1416
        range_y = 848     
        
    elif sceneName == "./annotations/hyang/video7/":
        range_x = 1450
        range_y = 1940
        
    elif sceneName == "./annotations/hyang/video8/" or sceneName == "./annotations/hyang/video9/":
        range_x = 1350
        range_y = 1940
        
    elif sceneName == "./annotations/hyang/video10/" or sceneName == "./annotations/hyang/video11/":
        range_x = 1416
        range_y = 748
        
    elif sceneName == "./annotations/hyang/video12/":
        range_x = 1316
        range_y = 848
        
    elif sceneName == "./annotations/hyang/video13/" or sceneName == "./annotations/hyang/video14/":
        range_x = 1316
        range_y = 748
        
    elif sceneName == "./annotations/little/video0/":
        range_x = 1417
        range_y = 2019
        
    elif sceneName == "./annotations/little/video1/" or sceneName == "./annotations/little/video2/":
        range_x = 1322
        range_y = 1945
        
    elif sceneName == "./annotations/little/video3/":
        range_x = 1422
        range_y = 1945
        
    elif sceneName == "./annotations/nexus/video0/" or sceneName == "./annotations/nexus/video2/":
        range_x = 1330
        range_y = 1947
        
    elif sceneName == "./annotations/nexus/video1/":
        range_x = 1430
        range_y = 1947
        
    elif sceneName == "./annotations/nexus/video3/" or sceneName == "./annotations/nexus/video5/":
        range_x = 1184
        range_y = 1759
        
    elif sceneName == "./annotations/nexus/video4/":
        range_x = 1284
        range_y = 1759
        
    elif sceneName == "./annotations/nexus/video6/" or sceneName == "./annotations/nexus/video8/":
        range_x = 1331
        range_y = 1962
        
    elif sceneName == "./annotations/nexus/video7/":
        range_x = 1431
        range_y = 1962
        
    elif sceneName == "./annotations/nexus/video9/":
        range_x = 1411
        range_y = 1980
        
        
    elif sceneName == "./annotations/nexus/video10/" or sceneName == "./annotations/nexus/video11/":
        range_x = 1311
        range_y = 1980
  
          
    elif sceneName == "./annotations/quad/video0/" or sceneName == "./annotations/quad/video1/" or sceneName == "./annotations/quad/video2/" or sceneName == "./annotations/quad/video3/":
        range_x = 1983
        range_y = 1088
            
        
    elif sceneName == "./annotations/bookstore/video0/":
        range_x = 1424
        range_y = 1088
    

    elif sceneName == "./annotations/bookstore/video1/" or sceneName == "./annotations/bookstore/video2/":
        range_x = 1422
        range_y = 1079

    elif sceneName == "./annotations/bookstore/video3/" or sceneName == "./annotations/bookstore/video4/" or sceneName == "./annotations/bookstore/video5/" or sceneName == "./annotations/bookstore/video6/":
        range_x = 1322
        range_y = 1079
        
    else:
        print(sceneName, ": Dataset not implemented yet!")
        
    """
    Section 1: Rearrange the resulting row format data into the order frame, pedestrianID, X-coordinate, Y-coordinate, compatible with ADI-LSTM script
    """
    
    csv = pd.read_csv((sceneName + inputFile), header = None, delimiter = r"\s+")

    ### If correcting the annotations, filter out all "lost"-labeled coordinates to prepare for later steps
    if vers == "corrected":
        csv = csv[csv[6]==0]

    ### If only interested in analyzing certain classes, can reintroduce this or similar lines.    
    #csv = csv[csv[9]=="Pedestrian"]
    csv[10] = csv[[1,3]].mean(axis = 1)
    csv[11] = csv[[2,4]].mean(axis = 1)
    csv = csv[[5, 0, 10, 11, 9]]

    csv.columns = range(csv.shape[1])
    rows = range(csv.shape[0])
    csv.index = rows

    csv.loc[:,2] = csv.loc[:,2].div(range_x)
    csv.loc[:,3] = csv.loc[:,3].div(range_y)
    
    csv.to_csv((sceneName + "pos_data_interp.csv"), header = False, index = False)
    
    return 0


i=0
### Hard-coded run type and the directory, could be improved for ease of use as a future update
# vers = "raw"
vers = "corrected"
for root, subdir, file in os.walk("./data/quad/" + vers):
  if len(file)>0:
      print(i)
      i+=1
        
      print("root: " + root)
      print("subdir: " + str(subdir))
      print("file: " + file[0])
      for f in file:
          if f == "annotations.txt":
              preprocess_raw((root+"/"), f, vers)
  