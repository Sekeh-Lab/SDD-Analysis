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
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def preprocess(sceneName, background, recordingMeta, tracksMeta, tracks):

    """
    Section 1: Rearrange the resulting row format data into the order frame, pedestrianID, X-coordinate, Y-coordinate
    """
    csv = pd.read_csv((sceneName + tracks))
    
    tracks = csv[["frame", "trackId", "xCenter", "yCenter"]]
    tracks.loc[:,"yCenter"] = tracks["yCenter"] * (-1)
    
    
    recMet = pd.read_csv((sceneName + recordingMeta))
    ptm = recMet.iloc[0,-1]
    
    
    img = plt.imread((sceneName + background))
    height = img.shape[0]
    width = img.shape[1]
    
    ### 12 is a scaling factor used by the author which seemed arbitrary except that it makes the data fit the image
    ### The result of this is to bound  annotations between 0 and 1 as with the SDD.
    tracks.loc[:,"xCenter"] = ((tracks["xCenter"] / (ptm)) / 12) / width
    tracks.loc[:,"yCenter"] = ((tracks["yCenter"] / (ptm)) / 12) / height

    ### Get class information from meta data files
    trkMeta = pd.read_csv((sceneName + tracksMeta))
    tracks["type"] = np.nan 
    for i in range(0, len(tracks.iloc[:,1].unique())):
        tracks.loc[tracks.loc[:,"trackId"] == i,"type"] = trkMeta.iloc[i,-1]


    csv.columns = range(csv.shape[1])
    rows = range(csv.shape[0])
    csv.index = rows

    ### The pos_data_interp.csv file here should have identical format to the ones in the SDD preprocessing
    tracks.to_csv((sceneName + "pos_data_interp.csv"), header = False, index = False)
    
    return 0



i=0
for root, subdir, file in os.walk("./"):
    if len(file)>0:
        print(i)
        i+=1
          
        print(root)
        print(subdir)
        print(file[0])
        directory = (root + "/")
        fileOne = ""
        fileTwo = ""
        fileThree = ""
        fileFour = ""
        for f in file:
            if f == "background.png":
                fileOne = "background.png"
            elif f == "recordingMeta.csv":
                fileTwo = "recordingMeta.csv"
            elif f == "tracksMeta.csv":
                fileThree = "tracksMeta.csv"
            elif f == "tracks.csv":
                fileFour = "tracks.csv"
        if fileOne == "background.png" and fileTwo == "recordingMeta.csv" and fileThree == "tracksMeta.csv" and fileFour == "tracks.csv":
            preprocess(directory, fileOne, fileTwo, fileThree, fileFour)
        else: 
            print(directory, ": Doesn't contain necessary input files")

