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
import numpy as np




def preprocess_raw(sceneName, inputFile):

    
    
    """
	Calculate the number of lost coordinates in the current video annotations in the beginning, middle, and end of all trajectories
    """
    start_lost = 0
    mid_lost = 0
    end_lost = 0
    
    csv = pd.read_csv((sceneName + inputFile), header = None, delimiter = r"\s+")
    
    #frame, id, class, lost
    csv = csv[[5, 0, 9, 6]]

    csv.columns = range(csv.shape[1])
    rows = range(csv.shape[0])
    csv.index = rows
    
    mask = np.ones((len(csv.iloc[:,0])), dtype = bool)

    d = int(csv.iloc[0,1])
    i = 0
    j = 0
    
    peds = csv.iloc[:,1].unique()


    for i in peds:
        traj = csv[csv[1].astype(int) == int(i)]
        lost = False
        mid_counted = False
        start_cap = False
        end_cap = False
        ### If current coordinate is labeled as lost
        if traj.iloc[0,3] == 1:
            lost = True
            start_lost += 1
            start_cap = True
        for n in range(0,len(traj)):
            if lost == True and traj.iloc[n,3] == 0:
                if start_cap == True:
                    start_cap = False
                    lost = False
                else:
                    end_cap = False
                    lost = False
                    if mid_counted == False:
                        mid_lost += 1
                        mid_counted = True

            elif lost == False and traj.iloc[n,3] == 1:
                lost = True
                end_cap = True
        if end_cap == True:
            end_lost += 1


    if not os.path.exists("./lostcounts/" + sceneName[2:]):
        os.makedirs("./lostcounts/" + sceneName[2:])


    metrics = pd.DataFrame(np.nan, index = [0], columns = ["peds", "start", "mid", "end"])
    metrics["peds"] = len(peds)
    metrics["start"] = start_lost
    metrics["mid"] = mid_lost
    metrics["end"] = end_lost


    metrics.to_csv(("./lostcounts/" + sceneName[2:] + "lostcounts.csv"), header = False, index = False)






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
            if f == "annotations.txt":
                preprocess_raw((root+"/"), f)
  