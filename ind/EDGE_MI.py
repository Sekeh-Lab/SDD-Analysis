
# EDGE Estimator for Shannon Mutual Information
#
# Created by Morteza Noshad (noshad@umich.edu)
# Current version: 4.3.1 
# Requirements: numpy, cvxpy(v1.0.6),scipy, sklearn
#                 
# 10/1/2018
#
# Based on the paper: Scalable Mutual Information Estimation using Dependence Graphs
#
################
# The estimator is in the following form:
#
# I = EDGE(X,Y,U=10, gamma=[1, 1], epsilon=[0,0], epsilon_vector = 'fixed', eps_range_factor=0.1, normalize_epsilon = False ,
#                ensemble_estimation = 'median', L_ensemble=5 ,hashing='p-stable', stochastic = False)
#
# Arguments: 
#
# X is N * d_x and Y is N * d_Y data sets
# U (optional) is an upper bound on the MI. It doesn't need to be accurate, but more accurate upper bound we set, faster convergence rates we get
# gamma=[gamma_X,gamma_Y] (optional) is the vector of soothness for X and Y. 
#        For example, if the data is discrete we set gamma close to 0, 
#        and if the data is continuous we set gamma close to 1 (or maybe higher if it is very smooth) 
# epsilon=[eps_X, eps_Y] (optional) is the vector of bandwidths for X and Y. If no epsilon is set, 
#        automatic bandwidths according to KNN distances will be set.
# epsilon_vector (optional): possible arguments are 'fixed' or 'range'. If 'fixed' is given, all of 
#        the bandwidths for the ensemble estimation will be the same, while, if 'range' is chosen, 
#        the badwidths will be arithmetically increasing in a range.     
# eps_range_factor (optional): If epsilon_vector == 'range', then the range of epsilon is 
#        [epsilon, epsilon*(1+epsilon_vector)].
# normalize_epsilon: If it is True, then the badwidth will be normalized according to the MI estimate 
# ensemble_estimation: several options are available:
#        'average': the ensemble estimator is the average of the base estimators
#        'optimal_weights': the ensemble estimator is the wighted sum of the base estimators
#                            where the weights are computed using an optimization problem
#                            * You need to import cvxpy as cvx (install cvxpy if you do not have it)
#        'median': the ensemble estimator is the median of the base estimators
# L_ensemble: number of different base estimators used in ensemble estimation. For more accurate estimates
#                you can increase L_ensemble, but runtime increases linearly as well.
# hashing (optional): possible arguments are 'p-stable' (default) which is a common type of LSH
#        or 'floor' which uses the simple floor function as hashing. For small dimensions, 'floor', a
#        for higher dimensions, 'p-stable' are preferred.
# stochastic: it is stochastic, the hashing is generated using a random seed.
# 
# Output: I is the estimation of mutual information between X snd Y 
###########################

import os
import numpy as np
import math
import cvxpy as cvx # Need to install CVXPY package, 
                    #  it is also possible to run this code without cvxpy, by 
                    #   using 'average' or 'median' ensemble_estimation
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
from datetime import datetime
import sys

#from random import randint, seed
#np.random.seed(seed=0)

#####################
#####################
# Find KNN distances for a number of samples for normalizing bandwidth






def find_knn(A,d):
    np.random.seed(3334)
    #np.random.seed()
    #np.random.seed(seed=int(time.time()))
    r = 500
    # random samples from A
    A = A.reshape((-1,1))
    N = A.shape[0]
    
    ### k is dependent on number of dimensions and frames, NOT values
    k=math.floor(0.43*N**(2/3 + 0.17*(d/(d+1)) )*math.exp(-1.0/np.max([10000, d**4])))
    #print('k,d', k, d)
    T= np.random.choice(A.reshape(-1,), size=r).reshape(-1,1)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(A)
    distances, indices = nbrs.kneighbors(T)
    d = np.mean(distances[:,-1])
    return d

# Returns epsilon and random shifts b
def gen_eps(XW,YV):
    d_X , d_Y  = XW.shape[1], YV.shape[1]
    # Find KNN distances for a number of samples for normalizing bandwidth
    eps_X = np.array([find_knn(XW[:,[i]],d_X) for i in range(d_X)]) + 0.0001
    eps_Y = np.array([find_knn(YV[:,[i]],d_Y) for i in range(d_Y)]) + 0.0001

    return (eps_X,eps_Y)

# Define H1 (LSH) for a vector X (X is just one sample)
def H1(XW,b,eps):
    X = XW
    # dimension of X
    d_X = XW.shape[0]
    #d_W = W.shape[1]
    XW=XW.reshape(1,d_X)

    # If not scalar
    if d_X > 1:
        X_te = 1.0*(np.squeeze(XW)+b)/eps    
    elif eps>0:
        X_te = 1.0*(XW+b)/eps
    else:
        X_te=XW

    # Discretize X
    X_t = np.floor(X_te)
    if d_X>1: 
        R = tuple(X_t.tolist())
    else: R=np.asscalar(np.squeeze(X_t))
    return R

# Compuate Hashing: Compute the number of collisions in each bucket
def Hash(XW,YV,eps_X,eps_Y,b_X,b_Y):

    # Num of Samples and dimensions
    N = XW.shape[0]

    # Hash vectors as dictionaries
    CX, CY, CXY = {}, {}, {} 
    
    # Computing Collisions
    
    for i in range(N):
        # Compute H_1 hashing of X_i and Y_i: Convert to tuple (vectors cannot be taken as keys in dict)

        X_l, Y_l = H1(XW[i],b_X,eps_X), H1(YV[i],b_Y,eps_Y)

        # X collisions: compute H_2 
        if X_l in CX:
            CX[X_l].append(i)
        else: 
            CX[X_l] = [i]
            
        # Y collisions: compute H_2
        if Y_l in CY:
            CY[Y_l].append(i)
        else: 
            CY[Y_l] = [i]

        # XY collisions
        if (X_l,Y_l) in CXY:
            CXY[(X_l,Y_l)].append(i)
        else: 
            CXY[(X_l,Y_l)] = [i]

    return (CX, CY, CXY)






def g_func(t):
    return((t-1)**2/(2*(t+1)))






def mi_t_edge(data, Tp = 20, U=10, gamma=[1, 1, 1], epsilon=[0,0,0], epsilon_vector = 'range', eps_range_factor=0.1, normalize_epsilon = True, ensemble_estimation = 'average', L_ensemble=10, hashing='p-stable', cmi='cmi3', ped_i = 'NA', ped_j ='NA'):


    ### Treats N as time and 2nd dimension as 2-tuple coordinates
    if(data.shape[0] == 2 and data[0].shape[0] == data[1].shape[0]):
        T = data[0].shape[0]
        I_T = 0;
        #print("start --------" + str(T))
        #time.sleep(1)

               
        if(T > Tp):
            
            MI_array = []
            for x in range(0,Tp):
                MI_array.append(0)
            ### Uses T' of 20 frames as a buffer for analysis

            for t in range(Tp, T):    
                x = data[0][:t , :].reshape(t, 2) 
                y = data[1][:t , :].reshape(t, 2) 
                
                MI = EDGE(x,y, hashing = hashing)
                I_value = max(0, MI)
                #print("     Time ", t)
                #print("     Value: ",I_value)
                MI_array.append(I_value)

            #print("Running ", T, " samples.")
            #print("Value is: ", I_T)
            

            return MI_array
        else:
            return -1
    else:
        return -1






def EDGE(X,Y, U=10, gamma=[1, 1], epsilon=[0,0], epsilon_vector = 'range', eps_range_factor=0.1, 
         normalize_epsilon = False, ensemble_estimation = 'average', L_ensemble=10 ,hashing='p-stable', stochastic = False):
    
    gamma = np.array(gamma)
    gamma = gamma * 0.4
    epsilon = np.array(epsilon)
    
    # Find dimensions
    dim_X, dim_Y  = X.shape[1], Y.shape[1]


## Hash type

    if hashing == 'floor':
        d_X_shrink, d_Y_shrink = dim_X, dim_Y 
        XW, YV = X, Y
    
## Initial epsilon and apply smoothness gamma

    # If no manual epsilon is set for computing MI:
    if epsilon[0] ==0:
        # Generate auto epsilon and b
        (eps_X_temp,eps_Y_temp) = gen_eps(XW,YV)
        # Normalizing factors for the bandwidths
        cx, cy = 18*d_X_shrink / np.max([(1+1.*math.log(dim_X)),1]), 18*d_Y_shrink/ np.max([(1+1.*math.log(dim_Y)),1])
        eps_X0, eps_Y0 = eps_X_temp * cx*gamma[0], eps_Y_temp * cy*gamma[1] 
        ##### At this point eps_X0 and Y0 have been derived from knn run on each dimension, and cx and cy only take into account
        ##### The number of dimensions, so it was identical for x and y, but since the knn differed slightly, eps is different still
    else:
        eps_X_temp = np.ones(d_X_shrink,)*epsilon[0]
        b_X = np.linspace(0,1,L_ensemble,endpoint=False)[j]*eps_X
        b_Y = np.linspace(0,1,L_ensemble,endpoint=False)[j]*eps_Y
        eps_Y_temp = np.ones(d_Y_shrink,)*epsilon[1]    
        cx, cy = 15*d_X_shrink / np.max([(1+1.0*math.log(dim_X)),1]), 15*d_Y_shrink/ np.max([(1+1.0*math.log(dim_Y)),1])
        eps_X0, eps_Y0 = eps_X_temp * cx*gamma[0], eps_Y_temp * cy*gamma[1] 

    ## epsilon_vector
    T = np.linspace(1,1+eps_range_factor,L_ensemble)        




## Compute MI Vector
    
    # MI Vector
    I_vec = np.zeros(L_ensemble)
    for j in range(L_ensemble):

        # Apply epsilon_vector 
        eps_X, eps_Y = eps_X0 * T[j], eps_Y0 * T[j]

        b_X = np.linspace(0,1,L_ensemble,endpoint=False)[j]*eps_X
        b_Y = np.linspace(0,1,L_ensemble,endpoint=False)[j]*eps_Y

        I_vec[j] = Compute_MI(XW,YV,U,eps_X,eps_Y,b_X,b_Y)

## Ensemble method
    if ensemble_estimation == 'average':
        I = np.mean(I_vec)
    elif ensemble_estimation == 'median':
        I = np.median(I_vec)

## Normalize epsilon according to MI estimation (cross validation)
    if normalize_epsilon == True:
        gamma=gamma * math.pow(2,-math.sqrt(I*2.0)+(0.5/I))
        normalize_epsilon = False
        I = EDGE(X,Y, U, gamma, epsilon, epsilon_vector, eps_range_factor, normalize_epsilon, ensemble_estimation, L_ensemble,hashing, stochastic)

    return I








# Compute mutual information and gradient given epsilons and radom shifts
def Compute_MI(XW,YV, U,eps_X,eps_Y,b_X,b_Y):
    N = XW.shape[0]

    (CX, CY, CXY) = Hash(XW,YV,eps_X,eps_Y,b_X,b_Y)
    
    # Computing Mutual Information Function
    I = 0
    J = 0
    N_c = 0
    # print(CXY)

            
    for e in CXY.keys():
        ### e is a 1x2x2 tensor, a pair of hashed coordinates
        Ni = len(CX[e[0]])
        Nj = len(CY[e[1]])
        Nij = len(CXY[e])
        
        I += (Ni/N)*(Nj/N)* g_func(Nij*N/(Ni*Nj))

    I = 1.0* I
    return I




def preprocess_EDGE(i, j, element, csv, frame_list):

    result = [0]
    
    if (j != i) and (element == 0):
        traj_data_i = csv[csv[1] == i]
        traj_data_i = traj_data_i.iloc[:,0:4]
        traj_data_i.index = range(traj_data_i.shape[0])
        
        ### Extract rows belonging to agents i and j, then reset the row names.
        traj_data_j = csv[csv[1] == j]
        traj_data_j = traj_data_j.iloc[:,0:4]

        traj_data_j.index = range(traj_data_j.shape[0])
        

        ### Prune rows belonging to frames that don't contain both agents
        traj_shared_i = traj_data_i[traj_data_i.iloc[:][0].isin(traj_data_j.iloc[:][0])]
        traj_shared_j = traj_data_j[traj_data_j.iloc[:][0].isin(traj_data_i.iloc[:][0])]
        ### Remove the frame and ped_ID columns, storing only the coordinate data
        frames = traj_shared_i[[0]]
        traj_shared_i = traj_shared_i[[2,3]]
        traj_shared_j = traj_shared_j[[2,3]]
        ### Store the pair of trajectory data for agents i and j
        traj_data = np.asarray([traj_shared_i.to_numpy(), traj_shared_j.to_numpy()])
        
        ### Append the MI to the dataframe instead, then extract rows with frame values located in obs, then convert
        frames = frames[20:]

        MI_traj = mi_t_edge(traj_data, hashing = 'floor', ped_i = str(i), ped_j = str(j))

        if(MI_traj!=-1):
            print("Agent j match ----------------", j, " length ", len(traj_data[0]), flush = True)

            frames['mi'] = MI_traj[20:]

            result = frames.values.tolist()

    #else:
        #print(element)
    return i, j, result


    
def parallelMI(sceneName, fileName, fileNameTwo, fileNameThree, normalized):
    print("started")
    csv = pd.read_csv((sceneName + fileName), header = None)
    csv_two = np.load((sceneName + fileNameTwo))
    csv_three = pd.read_csv((sceneName + fileNameThree), header = None)


    if sceneName == "./video0/" or sceneName == "./video1/" or sceneName == "./video2/" or sceneName == "./video3/":
        range_x = 1980
        range_y = 1093
        
    else:
        print(sceneName, ": Dataset not implemented yet!")
        

    if normalized == False:
        print(csv.iloc[0,2])
        csv.iloc[:,2] = csv.iloc[:,2]*range_x
        csv.iloc[:,3] = csv.iloc[:,3]*range_y
        print(csv.iloc[0,2])
    else:
        print("Normalized")

    frame_list = csv_three.iloc[0][:].unique()
    obs_peds = []
    
    for i in range(0,len(csv_two)):
        obs_peds.append(int(csv_two[i][0][0]))

    ped_num = int(max(csv.iloc[:][1])+1)
    print("a ", ped_num)

    ### Use numpy to populate 2D list of zeros, then convert to list so that variable lists can be stored
    MI_array = np.zeros((ped_num, ped_num))
    MI_array = MI_array.tolist()
    print("b")

    ped_range = list(range(0,ped_num))


    print("CPU count: ", 12)
    pool = mp.Pool(8)
    print("CPU count: ", mp.cpu_count())
    tuples = []   
    ### Haven't looked into how to pass the empty row by indexing into MI_array using obs_peds, but this should work
    for i in obs_peds:
        print("i is: " +str(i), flush = True)

        now = datetime.now()

        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
    
    
        tuples = pool.starmap_async(preprocess_EDGE, [(i, j, MI_array[i][j], csv, frame_list) for j in ped_range], chunksize = 1).get()
    

    
        for tup in tuples:
            if(tup[2] != [0]):
                if(MI_array[i][tup[1]]==0):
                    MI_array[i][tup[1]] = tup[2]
                if(MI_array[tup[1]][i]==0):
                    MI_array[tup[1]][i] = tup[2]
        
    pool.close()
    pool.join()

    
    MI_array = np.asarray(MI_array)
    print("done")
    if normalized == True:
        np.save((sceneName + 'MI_tensor.npy'), MI_array)
    else:
        np.save((sceneName + 'MI_tensor_fullres.npy'), MI_array)

####################################
####################################

if __name__ == "__main__":

    normalized = True
    i=0

    scene = " "
    scene = str(sys.argv[1])

    print ("argument is: ", int(sys.argv[1]))


    for root, subdir, file in os.walk("./" + scene):
        fileName = " "
        fileNameTwo = " "
        fileNameThree = " "

        if len(file)>0:
            print(i)
            i+=1
            
            print(root)
            directory = (root +"/")

            for f in file:
                if f == "pos_data_interp.csv":
                    fileName = (f)
                
                elif f == "obs.npy":
                    fileNameTwo = (f)
                    
                elif f == "pos_data.csv":
                    fileNameThree = (f)
                    
            print("Check")
            if fileName == "pos_data_interp.csv" and fileNameTwo == "obs.npy" and fileNameThree == "pos_data.csv":
                print("passed")
                parallelMI(directory, fileName, fileNameTwo, fileNameThree, normalized)
            else: 
                print(directory, ": Doesn't contain necessary input files for MI calculation")

            
                