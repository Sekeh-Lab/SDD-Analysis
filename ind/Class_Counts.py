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





    """
	Counts the number of occurences of each class label in each video	
    """
         
def preprocess(sceneName, inputFile):
   

    csv = pd.read_csv((sceneName + inputFile), header = None, delimiter = r',')

    count_ped = 0
    count_bike = 0
    count_car = 0
    count_truckbus = 0
    count_misc = 0

    for i in len(csv):
        if csv.iloc[i,7] == "pedestrian":
            count_ped += 1
        elif csv.iloc[i,7] == "bicycle":
            count_bike += 1
        elif csv.iloc[i,7] == "car":
            count_car += 1
        elif csv.iloc[i,7] == "truck_bus":
            count_truckbus += 1
        else:
            count_misc += 1

    metrics = pd.DataFrame(np.nan, index = [0], columns = ["peds", "bikes", "cars", "truck_buses", "other"])
    metrics.iloc[0,0] = count_ped
    metrics.iloc[0,1] = count_bike
    metrics.iloc[0,2] = count_car
    metrics.iloc[0,3] = count_truckbus
    metrics.iloc[0,4] = count_misc


    csv.to_csv((sceneName + "counts.csv"), header = False, index = False)
                
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
            if f == "tracksMeta.csv":
                preprocess((root+"/"), f)
  