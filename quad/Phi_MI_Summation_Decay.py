import os
import numpy as np
import math
import time
from scipy.special import *
from sklearn.neighbors import NearestNeighbors
import sklearn
from csv import reader
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import multiprocessing as mp
import itertools
import copy
import sys
#####################
#####################

"""
Use the previously calculated MI and Phi values to calculate AIM
"""

####################################
####################################
def summation(sceneName, frame_File, mi_File, phi_File, phi_type, relative = False, decay = 0):
    print("Decay: ", decay)
    csv = pd.read_csv((sceneName + frame_File), header = None)

    frame_list = csv.iloc[0][:].unique()
    # print(len(frame_list), flush = True)
    # print(frame_list)
    mi_array = np.load((sceneName + mi_File), allow_pickle = True)
    phi_array = np.load((sceneName + phi_File), allow_pickle = True)
    AIM_array = np.zeros((len(mi_array),len(mi_array[0])))


    mi_array = mi_array.tolist()
    phi_array = phi_array.tolist()    
    AIM_array = AIM_array.tolist()

    ### Create a blank AIM_array in the variable 4D shape of mi_array
    for i in range(len(mi_array)):
        for j in range(len(mi_array[0])):
            if phi_array[i][j] != 0:
                AIM_array[i][j] = np.zeros((len(phi_array[i][j]),2)).tolist()
            

    

    
    ### Apply phi and sum MI values
    for i in range(len(mi_array)):
        for j in range(len(mi_array[0])):
            if phi_array[i][j] != 0 and mi_array[i][j] != 0:
                x = mi_array[i][j]
                if relative == True:
                    y = phi_array[i][j][1:]
                else:
                    y = phi_array[i][j]
                    
                AIM = 0
                ### Across all frames in a trajectory, if the frames are the same at each time index, sum phi*MI into the AIM
                ind = 0
                if mi_array[i][j][0][0] <= y[0][0]:
                    for n in range(len(mi_array[i][j])):
                        if ind < len(y):
                            if mi_array[i][j][n][0] == y[ind][0]:
                                AIM = (AIM*(1-decay)) + (mi_array[i][j][n][1] * y[ind][1])
                                #AIM += (mi_array[i][j][n][1])
                                AIM_array[i][j][ind][0] = int(mi_array[i][j][n][0])
                                AIM_array[i][j][ind][1] = AIM
                                ind += 1
                else:
                    print(sceneName)
                    print("Error, phi starts before MI!")
                # for n in range(len(mi_array[i][j])):
                #     if mi_array[i][j][n][0] == y[n][0]:
                #         AIM = (AIM*(1-decay)) + (mi_array[i][j][n][1] * y[n][1])
                #         AIM_array[i][j][ind][0] = int(mi_array[i][j][n][0])
                #         AIM_array[i][j][ind][1] = AIM
                #     else:
                #         print(sceneName)
                #         print("Error, frames mismatched!")
                # arr = AIM_array[i][j]
                
                ### Alternatively find a way to do this with pandas and use isin()
                
                # matched = []
                # for tup in AIM_array[i][j]:
                #     for frame in frame_list:
                #         if tup[0] == int(frame):
                #             matched.append(tup)
                # AIM_array[i][j] = matched

            #result = frames[frames.iloc[:][0].isin(frame_list)].values.tolist()


    print(AIM_array[4][5])
        
    AIM_array = np.asarray(AIM_array)
    np.save((sceneName + 'aim/AIM_summed_' + phi_type + '_Decay_{}.npy'.format(str(decay*100))), AIM_array)






phi_type = "03v_hd"

phifile = ("phi/phi_tensor_" + phi_type + ".npy")

### Supply the annotation processing method as a command line argument
vers = ""
if sys.argv[1] == "raw":
    vers = "raw"
elif sys.argv[1] == "corrected":
    vers = "corrected"

### Calculate AIM for a set of different decay values
# for dec in [0, .01, 0.05, 0.1]:
for dec in [0.005, 0.02, 0.03]:
    i=0
       
    for root, subdir, file in os.walk("./" + vers):
        if len(file)>0:
            print(i)
            i+=1
            
            print(root)
            print(subdir)
            print(file)
            directory = (root +"/")
            frame_File = " "
            mi_File = " "
            phi_File = " "
            for f in file:
                if f == "pos_data.csv":
                    frame_File = (f)

                    if os.path.isfile(directory + "mi/MI_tensor.npy"):
                        mi_File = "mi/MI_tensor.npy"
                        
                    if os.path.isfile(directory + phifile):
                        phi_File = phifile
                if frame_File == "pos_data.csv" and mi_File == "mi/MI_tensor.npy" and phi_File == phifile:
                    summation(directory, frame_File, mi_File, phi_File, phi_type, False, dec)
                else: 
                    print(directory, ": Doesn't contain necessary input files for AIM calculation")