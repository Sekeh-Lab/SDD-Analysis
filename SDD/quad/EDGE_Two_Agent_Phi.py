
import os
import numpy as np
import math
import cvxpy as cvx
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
import sys


### suppress warnings for pandas caused by copying without loc
pd.options.mode.chained_assignment = None


### Phi parameter calculations    
def calc_dvac(XW, YV, N, agent_type = "pedestrian"):
    T = XW.shape[0]
    hN = N
    t = T-hN
    t2 = T-N
    ############################ Velocity ###################################

    sigma = 0
    vel_x = np.zeros(hN)
    vel_y = np.zeros(hN)
    n = 0
    ### This is never called when i=0, due to our approach of using T_prime as a buffer window
    for i in range(t,T):
        ### Distance between XW[i] and the previous point of XW[i-1]. In order to use velocity for subsequent calculations,  
        ###     Im keeping the units in terms of distance/frame, rather than distance/second
        
        vel_x[n] = ((XW[i][0]-XW[i-1][0])**2 + (XW[i][1]-XW[i-1][1])**2)**(1/2)
        vel_y[n] = ((YV[i][0]-YV[i-1][0])**2 + (YV[i][1]-YV[i-1][1])**2)**(1/2)
        sigma +=  vel_x[n] + vel_y[n]
        n += 1
    velocity = sigma/hN
    vel_norm = 1-(1/(np.exp(np.log(500*velocity))+1))


    ############################ Acceleration ###################################

    sigma = 0   
    for i in range(1, len(vel_x)):
        ### using acceleration = d.velocity/d.time, time is in units of frames rather than seconds
        acc_x = abs(vel_x[i] - vel_x[i-1])
        acc_y = abs(vel_y[i] - vel_y[i-1])
        sigma += acc_x + acc_y
    acceleration = sigma/(hN-1) 

    ### Attempt to normalize the acceleration term according to velocity and a tanh function to roughly capture the relative change in velocity
    #acceleration = (2/(1+np.exp(-acceleration/velocity)))-1
    if velocity != 0:
        ### Sigmoidal function ranging from ~(0,0) to ~(1,1), centered on (0.5, 0.5)
        acceleration = 1-np.exp(-(350*acceleration)**2)
    else:
        acceleration = 0
    
    


    ############################ Distance ###################################

    sigma_one = 0
    sigma_two = 0
    dist_array = np.zeros([hN+1])
    n = 0
    ### Gets average distance between i and j over window N
    for i in range(t-1,T):
        dist_array[n] = ((XW[i][0]-YV[i][0])**2 + (XW[i][1]-YV[i][1])**2)**(1/2)
        if i == t-1:
            sigma_one += dist_array[n]
        else:
            sigma_one += dist_array[n]
            sigma_two += dist_array[n] 
        n += 1
    distance = sigma_two/hN

    
    ### This maps a sigmoidal function with domain [-inf, inf] and range [0,2] (roughly a mirrored tanh function), passing through (0,1)
    ### Since all distances are restricted to positive values, this outputs between 0 and 1, decreasing diminishingly with distance
    ### Im not sure if it makes sense to do it this way, if it really needs to be bound to a fraction, or if values > 1 make sense
    #distance = (2/(1+np.exp(distance)))
    if agent_type == "pedestrian":
        distance = np.exp(-5*(distance**2))
    elif agent_type == "vehicle":
        distance = np.exp(-10*(distance**4))
    else:
        print("agent type not implemented: ", agent_type)
    


    ### if the agents are moving directly toward eachother, the sum of their velocities would equal the rate at which they converge,
    ###     In this case, convergence/velocities should normalize to 1 through an activation function.
    ### If the agents are moving parallel to eachother, convergence is 0 and velocity doesn"t matter in convergence/velocities. 
    ###     In this case, 0/velocities should normalize to 0
    ### If the agents are moving away from eachother, the term convergence/velocities would be negative
    ###     In this case, -convergence/velocities could either be normalized to a negative value (and an activation function over phi accounts for it)
    ###         or as with ReLU activation functions, negative values could be set to 0. I prefer the first option and will go with it for now
    
    ### Rather than an absolute value, the sign of conv is important to signify whether the agents are converging or diverging

    ############################ Convergence ###################################

    sigma = 0
    for i in range(1, len(dist_array)):
        ### The change in distance between each frame in the window N
        conv = dist_array[i] - dist_array[i-1]
        sigma += conv
    convergence = sigma/(hN)


    if velocity !=0:
        convergence = (-1.0) * convergence/velocity
    else: 
        convergence = 0



    ############################ Heading ###################################

    heading_array_x = np.empty([N])
    heading_array_y = np.empty([N])
    n = 0
    ### Gets average distance between i and j over window N
    for i in range(t2,T):
        px0 = [XW[i-1][0], XW[i-1][1]]
        px1 = [XW[i][0], XW[i][1]]
        py0 = [YV[i-1][0], YV[i-1][1]]
        py1 = [YV[i][0], YV[i][1]]

        ### Lines for calculating x"s visibility of y
        vx0 = np.array(px1) - np.array(px0)
        vx1 = np.array(py0) - np.array(px0)

        ### Lines for calculating y"s visibility of x
        vy0 = np.array(py1) - np.array(py0)
        vy1 = np.array(px0) - np.array(py0)

        if (vx0[0] == 0 and vx0[1] == 0) or (vx1[0] == 0 and vx1[1] == 0):
            heading_array_x[n] = np.NaN
        else:
            heading_array_x[n] = abs(np.math.atan2(np.linalg.det([vx0, vx1]), np.dot(vx0, vx1)))
    

        if (vy0[0] == 0 and vy0[1] == 0) or (vy1[0] == 0 and vy1[1] == 0):
            heading_array_y[n] = np.NaN
        else:
            heading_array_y[n] = abs(np.math.atan2(np.linalg.det([vy0, vy1]), np.dot(vy0, vy1)))

        n += 1
    ### Base visibility off of the best recent heading for 
    ###    detection (even if the other agent moves away, this reflects the likelihood that theyve been seen)

    heading_x = np.nanmean(heading_array_x)
    heading_y = np.nanmean(heading_array_y)

    if np.isnan(heading_x):
        heading_x = (math.pi/2)
    if np.isnan(heading_y):
        heading_y = (math.pi/2)



    heading_x = -math.sin(heading_x-(math.pi/2))
    heading_y = -math.sin(heading_y-(math.pi/2))



    ############################ Delta Heading ###################################
    delta_heading_x = 0
    delta_heading_y = 0
    for i in range(t2,T):
        ### Calculate the angle for each agent between their headings at the start and stop of the last frame
        px0 = [XW[i-2][0], XW[i-2][1]]
        px1 = [XW[i-1][0], XW[i-1][1]]
        px2 = [XW[i][0], XW[i][1]]

        py0 = [YV[i-2][0], YV[i-2][1]]
        py1 = [YV[i-1][0], YV[i-1][1]]
        py2 = [YV[i][0], YV[i][1]]

        ### Compute the previous and current vector for x
        vx0 = np.array(px1) - np.array(px0)
        vx1 = np.array(px2) - np.array(px1)

        ### Compute the previous and current vector for y
        vy0 = np.array(py1) - np.array(py0)
        vy1 = np.array(py2) - np.array(py1)

        if (abs(vx0[0]) < 0.001 and abs(vx0[1]) < 0.001) or (abs(vx1[0]) < 0.001 and abs(vx1[1]) < 0.001):
            delta_heading_x +=  0 
        else: 
            delta_heading_x += abs(np.math.atan2(np.linalg.det([vx0, vx1]), np.dot(vx0, vx1)))

        if (abs(vy0[0]) < 0.001 and abs(vy0[1]) < 0.001) or (abs(vy1[0]) < 0.001 and abs(vy1[1]) < 0.001):
            delta_heading_y +=  0 
        else:
            delta_heading_y += abs(np.math.atan2(np.linalg.det([vy0, vy1]), np.dot(vy0, vy1)))

    delta_heading_x = -(1/(1+np.exp(3*(delta_heading_x-((N*math.pi)/6)))))+1
    delta_heading_y = -(1/(1+np.exp(3*(delta_heading_y-((N*math.pi)/6)))))+1


    return vel_norm, acceleration, distance, convergence, heading_x, heading_y, delta_heading_x, delta_heading_y

   




def mi_t_edge(data, agent_type = "", phi_type = "", Tp = 60,  hashing="p-stable", ped_i = "NA", ped_j ="NA"):


    ### Treats N as time and 2nd dimension as 2-tuple coordinates
    if(data.shape[0] == 2 and data[0].shape[0] == data[1].shape[0]):
        T = data[0].shape[0]
        I_T = 0;


               
        if(T > Tp):
            print("start ----------------" + str(T))
            
            phi_array_x = []
            phi_array_y = []
            for x in range(0,Tp):
                phi_array_x.append(0)
                phi_array_y.append(0)
            ### Uses T" of 20 frames as a buffer for analysis

            for t in range(Tp, T):    
                x = data[0][:t , :].reshape(t, 2) 
                y = data[1][:t , :].reshape(t, 2) 
                
                phi_x, phi_y = get_phi(x,y, agent_type = agent_type, phi_type = phi_type, hashing = hashing)
                #print("     Time ", t)
                #print("     Value: ",I_value)
                phi_array_x.append(phi_x)
                phi_array_y.append(phi_y)

            #print("Running ", T, " samples.")
            #print("Value is: ", I_T)
            

          
            return phi_array_x, phi_array_y
        else:
            #print("Insufficient time points for T prime")  
            return -1, -1
    else:
        #print("Incompatible data structure, supply a 2xNx2 tensor")
        return -1, -1






def get_phi(X,Y, agent_type = "pedestrian", phi_type = "", hashing="floor"):
    if hashing == "floor":
        XW, YV = X, Y
    
    # W = 5
    # W = 20
    # W = 60
    W = 30




"""
   This section contains numerous different setups which were tested for the phi function, to determine which one most sensibly
        reflected a human-like expectation of how attention would be paid during an interaction.
"""

    if phi_type == "vga_05hdc":
        
        ### vel: 0 to inf;  acc: 0 to 1; dis: 1 to 0; conv: -1 to 1
        if agent_type == "pedestrian":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "pedestrian")
            a = 0.5
            b = 3
            c = 0.3
        elif agent_type == "vehicle":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "vehicle")
            a = 0
            b = 3
            c = 0.3
        phi_x = (c + vel * (1 + dhead_x + acc)) * (a + (dis * (1 + head_x)) + conv)
        phi_y = (c + vel * (1 + dhead_y + acc)) * (a + (dis * (1 + head_y)) + conv)



    elif phi_type == "ga_05hdc":
        
        ### vel: 0 to inf;  acc: 0 to 1; dis: 1 to 0; conv: -1 to 1
        if agent_type == "pedestrian":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "pedestrian")
            a = 0.5
            b = 3
            c = 0.3
        elif agent_type == "vehicle":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "vehicle")
            a = 0
            b = 3
            c = 0.3

        phi_x = (c + dhead_x + acc) * (a + (dis * (1 + head_x)) + conv)
        phi_y = (c + dhead_y + acc) * (a + (dis * (1 + head_y)) + conv)


    elif phi_type == "va_05hdc":
        
        ### vel: 0 to inf;  acc: 0 to 1; dis: 1 to 0; conv: -1 to 1
        if agent_type == "pedestrian":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "pedestrian")
            a = 0.5
            b = 3
            c = 0.3
        elif agent_type == "vehicle":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "vehicle")
            a = 0
            b = 3
            c = 0.3

        phi_x = (c + vel * (1 + acc)) * (a + (dis * (1 + head_x)) + conv)
        phi_y = (c + vel * (1 + acc)) * (a + (dis * (1 + head_y)) + conv)

    elif phi_type == "vg_05hdc":
        
        ### vel: 0 to inf;  acc: 0 to 1; dis: 1 to 0; conv: -1 to 1
        if agent_type == "pedestrian":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "pedestrian")
            a = 0.5
            b = 3
            c = 0.3
        elif agent_type == "vehicle":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "vehicle")
            a = 0
            b = 3
            c = 0.3

        phi_x = (c + vel * (1 + dhead_x)) * (a + (dis * (1 + head_x)) + conv)
        phi_y = (c + vel * (1 + dhead_y)) * (a + (dis * (1 + head_y)) + conv)



    elif phi_type == "vga_05dc":
        
        ### vel: 0 to inf;  acc: 0 to 1; dis: 1 to 0; conv: -1 to 1
        if agent_type == "pedestrian":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "pedestrian")
            a = 0.5
            b = 3
            c = 0.3
        elif agent_type == "vehicle":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "vehicle")
            a = 0
            b = 3
            c = 0.3

        phi_x = (c + vel * (1 + dhead_x + acc)) * (a + dis + conv)
        phi_y = (c + vel * (1 + dhead_y + acc)) * (a + dis + conv)


    elif phi_type == "vga_05hc":
        
        ### vel: 0 to inf;  acc: 0 to 1; dis: 1 to 0; conv: -1 to 1
        if agent_type == "pedestrian":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "pedestrian")
            a = 0.5
            b = 3
            c = 0.3
        elif agent_type == "vehicle":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "vehicle")
            a = 0
            b = 3
            c = 0.3

        phi_x = (c + vel * (1 + dhead_x + acc)) * (a + (1 + head_x)/2 + conv)
        phi_y = (c + vel * (1 + dhead_y + acc)) * (a + (1 + head_y)/2 + conv)


    elif phi_type == "vga_05hd":
        
        ### vel: 0 to inf;  acc: 0 to 1; dis: 1 to 0; conv: -1 to 1
        if agent_type == "pedestrian":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "pedestrian")
            a = 0.5
            b = 3
            c = 0.3
        elif agent_type == "vehicle":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "vehicle")
            a = 0
            b = 3
            c = 0.3

        phi_x = (c + vel * (1 + dhead_x + acc)) * (a + (dis * (1 + head_x)))
        phi_y = (c + vel * (1 + dhead_y + acc)) * (a + (dis * (1 + head_y)))

################################ 4 parameters ####################################################
    elif phi_type == "a_05hdc":
        
        ### vel: 0 to inf;  acc: 0 to 1; dis: 1 to 0; conv: -1 to 1
        if agent_type == "pedestrian":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "pedestrian")
            a = 0.5
            b = 3
            c = 0.3
        elif agent_type == "vehicle":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "vehicle")
            a = 0
            b = 3
            c = 0.3
        phi_x = (c + acc) * (a + (dis * (1 + head_x)) + conv)
        phi_y = (c + acc) * (a + (dis * (1 + head_y)) + conv)



    elif phi_type == "v_05hdc":
        
        ### vel: 0 to inf;  acc: 0 to 1; dis: 1 to 0; conv: -1 to 1
        if agent_type == "pedestrian":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "pedestrian")
            a = 0.5
            b = 3
            c = 0.3
        elif agent_type == "vehicle":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "vehicle")
            a = 0
            b = 3
            c = 0.3

        phi_x = (c + vel) * (a + (dis * (1 + head_x)) + conv)
        phi_y = (c + vel) * (a + (dis * (1 + head_y)) + conv)


    elif phi_type == "va_05dc":
        
        ### vel: 0 to inf;  acc: 0 to 1; dis: 1 to 0; conv: -1 to 1
        if agent_type == "pedestrian":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "pedestrian")
            a = 0.5
            b = 3
            c = 0.3
        elif agent_type == "vehicle":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "vehicle")
            a = 0
            b = 3
            c = 0.3

        phi_x = (c + vel * (1 + acc)) * (a + (dis) + conv)
        phi_y = (c + vel * (1 + acc)) * (a + (dis) + conv)



    elif phi_type == "va_05hc":
        
        ### vel: 0 to inf;  acc: 0 to 1; dis: 1 to 0; conv: -1 to 1
        if agent_type == "pedestrian":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "pedestrian")
            a = 0.5
            b = 3
            c = 0.3
        elif agent_type == "vehicle":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "vehicle")
            a = 0
            b = 3
            c = 0.3

        phi_x = (c + vel * (1 + acc)) * (a + (1 + head_x)/2 + conv)
        phi_y = (c + vel * (1 + acc)) * (a + (1 + head_y)/2 + conv)


    elif phi_type == "va_05hd":
        
        ### vel: 0 to inf;  acc: 0 to 1; dis: 1 to 0; conv: -1 to 1
        if agent_type == "pedestrian":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "pedestrian")
            a = 0.5
            b = 3
            c = 0.3
        elif agent_type == "vehicle":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "vehicle")
            a = 0
            b = 3
            c = 0.3

        phi_x = (c + vel * (1 + acc)) * (a + (dis * (1 + head_x)))
        phi_y = (c + vel * (1 + acc)) * (a + (dis * (1 + head_y)))


################################ 3 parameters ####################################################
    elif phi_type == "a_05hd":
        
        ### vel: 0 to inf;  acc: 0 to 1; dis: 1 to 0; conv: -1 to 1
        if agent_type == "pedestrian":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "pedestrian")
            a = 0.5
            b = 3
            c = 0.3
        elif agent_type == "vehicle":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "vehicle")
            a = 0
            b = 3
            c = 0.3

        phi_x = (c + acc) * (a + (dis * (1 + head_x)))
        phi_y = (c + acc) * (a + (dis * (1 + head_y)))

    elif phi_type == "v_05hd":
        
        ### vel: 0 to inf;  acc: 0 to 1; dis: 1 to 0; conv: -1 to 1
        if agent_type == "pedestrian":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "pedestrian")
            a = 0.5
            b = 3
            c = 0.3
        elif agent_type == "vehicle":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "vehicle")
            a = 0
            b = 3
            c = 0.3

        phi_x = (c + vel) * (a + (dis * (1 + head_x)))
        phi_y = (c + vel) * (a + (dis * (1 + head_y)))

    elif phi_type == "03v_hd":
        
        ### vel: 0 to inf;  acc: 0 to 1; dis: 1 to 0; conv: -1 to 1
        if agent_type == "pedestrian":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "pedestrian")
            a = 0
            b = 3
            c = 0.3
        elif agent_type == "vehicle":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "vehicle")
            a = 0
            b = 3
            c = 0.3

        phi_x = (c + vel) * (a + (dis * (1 + head_x)))
        phi_y = (c + vel) * (a + (dis * (1 + head_y)))

    elif phi_type == "va_05d":
        
        ### vel: 0 to inf;  acc: 0 to 1; dis: 1 to 0; conv: -1 to 1
        if agent_type == "pedestrian":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "pedestrian")
            a = 0.5
            b = 3
            c = 0.3
        elif agent_type == "vehicle":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "vehicle")
            a = 0
            b = 3
            c = 0.3

        phi_x = (c + vel * (1 + acc)) * (a + (dis))
        phi_y = (c + vel * (1 + acc)) * (a + (dis))


    elif phi_type == "va_05h":
        
        ### vel: 0 to inf;  acc: 0 to 1; dis: 1 to 0; conv: -1 to 1
        if agent_type == "pedestrian":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "pedestrian")
            a = 0.5
            b = 3
            c = 0.3
        elif agent_type == "vehicle":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "vehicle")
            a = 0
            b = 3
            c = 0.3

        phi_x = (c + vel * (1 + acc)) * (a + (1 + head_x)/2)
        phi_y = (c + vel * (1 + acc)) * (a + (1 + head_y)/2)
################################ 2 or 1 parameters ####################################################



    elif phi_type == "03_hd":
        
        ### vel: 0 to inf;  acc: 0 to 1; dis: 1 to 0; conv: -1 to 1
        if agent_type == "pedestrian":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "pedestrian")
            a = 0
            b = 3
            c = 0.3
        elif agent_type == "vehicle":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "vehicle")
            a = 0
            b = 3
            c = 0.3

        phi_x = (c) * (a + (dis * (1 + head_x)))
        phi_y = (c) * (a + (dis * (1 + head_y)))



    elif phi_type == "03v_d":
        
        ### vel: 0 to inf;  acc: 0 to 1; dis: 1 to 0; conv: -1 to 1
        if agent_type == "pedestrian":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "pedestrian")
            a = 0
            b = 3
            c = 0.3
        elif agent_type == "vehicle":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "vehicle")
            a = 0
            b = 3
            c = 0.3

        phi_x = (c + vel) * (a + (dis))
        phi_y = (c + vel) * (a + (dis))




    elif phi_type == "03v_h":
        
        ### vel: 0 to inf;  acc: 0 to 1; dis: 1 to 0; conv: -1 to 1
        if agent_type == "pedestrian":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "pedestrian")
            a = 0
            b = 3
            c = 0.3
        elif agent_type == "vehicle":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "vehicle")
            a = 0
            b = 3
            c = 0.3

        phi_x = (c + vel) * (a + ((1 + head_x)/2))
        phi_y = (c + vel) * (a + ((1 + head_y)/2))




    elif phi_type == "v":
        
        ### vel: 0 to inf;  acc: 0 to 1; dis: 1 to 0; conv: -1 to 1
        if agent_type == "pedestrian":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "pedestrian")
            b = 3
            c = 0.3
        elif agent_type == "vehicle":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "vehicle")
            b = 3
            c = 0.3

        phi_x =  (vel)
        phi_y = phi_x

    elif phi_type == "a":
        
        ### vel: 0 to inf;  acc: 0 to 1; dis: 1 to 0; conv: -1 to 1
        if agent_type == "pedestrian":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "pedestrian")
            b = 3
            c = 0.3
        elif agent_type == "vehicle":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "vehicle")
            b = 3
            c = 0.3

        phi_x =  (acc)
        phi_y = phi_x

    elif phi_type == "d":
        
        ### vel: 0 to inf;  acc: 0 to 1; dis: 1 to 0; conv: -1 to 1
        if agent_type == "pedestrian":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "pedestrian")
            b = 3
            c = 0.3
        elif agent_type == "vehicle":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "vehicle")
            b = 3
            c = 0.3

        phi_x =  (dis)
        phi_y = phi_x

    elif phi_type == "c":
        
        ### vel: 0 to inf;  acc: 0 to 1; dis: 1 to 0; conv: -1 to 1
        if agent_type == "pedestrian":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "pedestrian")
            b = 3
            c = 0.3
        elif agent_type == "vehicle":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "vehicle")
            b = 3
            c = 0.3

        phi_x =  conv
        phi_y = phi_x


    elif phi_type == "g":
        
        ### vel: 0 to inf;  acc: 0 to 1; dis: 1 to 0; conv: -1 to 1
        if agent_type == "pedestrian":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "pedestrian")
            b = 3
            c = 0.3
        elif agent_type == "vehicle":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "vehicle")
            b = 3
            c = 0.3

        phi_x = dhead_x
        phi_y = dhead_y




    elif phi_type == "h":
        
        ### vel: 0 to inf;  acc: 0 to 1; dis: 1 to 0; conv: -1 to 1
        if agent_type == "pedestrian":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "pedestrian")
            a = 0.5
            b = 3
            c = 0.3
        elif agent_type == "vehicle":
            vel, acc, dis, conv, head_x, head_y, dhead_x, dhead_y = calc_dvac(XW, YV, W, agent_type = "vehicle")
            a = 0
            b = 3
            c = 0.3

        phi_x = head_x
        phi_y = head_y


    else:
        phi_x = 1.
        phi_y = 1.

    # phi_x = max(0, (2/(1+np.exp(-b*phi_x)))-1)
    # phi_y = max(0, (2/(1+np.exp(-b*phi_y)))-1)
    phi_x = max(0, phi_x)
    phi_y = max(0, phi_y)

    return phi_x, phi_y




def preprocess_EDGE(i, j, element_x, element_y, csv, frame_list, phi_type):
    Tp = 30
    result_x = [0]
    result_y = [0]
    check = []
    if (j != i) and (element_x == 0 or element_y == 0):
        label_i = ""
        label_j = ""
        traj_data_i = csv[csv[1] == i]
        traj_data_i.index = range(traj_data_i.shape[0])
        
        ### Extract rows belonging to agents i and j, then reset the row names.
        traj_data_j = csv[csv[1] == j]
        traj_data_j.index = range(traj_data_j.shape[0])
        check = traj_data_j
        
        
        if len(check) > 0:
            ### Prune rows belonging to frames that don"t contain both agents
            traj_shared_i = traj_data_i[traj_data_i.iloc[:][0].isin(traj_data_j.iloc[:][0])]
            traj_shared_j = traj_data_j[traj_data_j.iloc[:][0].isin(traj_data_i.iloc[:][0])]
            ### Remove the frame and ped_ID columns, storing only the coordinate data
            frames = traj_shared_i[[0]]
            traj_shared_i = traj_shared_i[[2,3]]
            traj_shared_j = traj_shared_j[[2,3]]
            ### Store the pair of trajectory data for agents i and j
            traj_data = np.asarray([traj_shared_i.to_numpy(), traj_shared_j.to_numpy()])
            
            
            ### Append the ADI to the dataframe instead, then extract rows with frame values located in obs, then convert
            frames_x = frames[Tp:]
            frames_y = frames[Tp:]

            if traj_data_i.iloc[0][4] == "Pedestrian" or traj_data_i.iloc[0][4] == "Skater" or traj_data_i.iloc[0][4] == "Biker":
                label_i = "Pedestrian"
            elif traj_data_i.iloc[0][4] == "Cart" or traj_data_i.iloc[0][4] == "Bus" or traj_data_i.iloc[0][4] == "Car":
                label_i = "Vehicle"
            else:
                print("Unrecognized Agent Class for: ", traj_data_i.iloc[0][0])
                
            if traj_data_j.iloc[0][4] == "Pedestrian" or traj_data_j.iloc[0][4] == "Skater" or traj_data_j.iloc[0][4] == "Biker":
                label_j = "Pedestrian"
            elif traj_data_j.iloc[0][4] == "Cart" or traj_data_j.iloc[0][4] == "Bus" or traj_data_j.iloc[0][4] == "Car":
                label_j = "Vehicle"
            else:
                print("Unrecognized Agent Class for: ", traj_data_j.iloc[0][0])
                
            if label_i == "Pedestrian" and label_j == "Pedestrian":
                phi_traj_x, phi_traj_y = mi_t_edge(traj_data, agent_type = "pedestrian", phi_type = phi_type, Tp = Tp, hashing = "floor", ped_i = str(i), ped_j = str(j))
            elif label_i == "Vehicle" or label_j == "Vehicle":
                phi_traj_x, phi_traj_y = mi_t_edge(traj_data, agent_type = "vehicle", phi_type = phi_type, Tp = Tp, hashing = "floor", ped_i = str(i), ped_j = str(j))
            else:
                print("Error, unsupported label!")
                
                
            if(phi_traj_x!=-1):
                frames_x["phi"] = phi_traj_x[Tp:]
                result_x = frames_x.values.tolist()

            if(phi_traj_y!=-1):
                frames_y["phi"] = phi_traj_y[Tp:]
                result_y = frames_y.values.tolist()

    return i, j, result_x, result_y


    
def parallelPhi(sceneName, fileName, fileNameTwo, fileNameThree):
    print("started")
    csv = pd.read_csv((sceneName + fileName), header = None)
    csv_two = np.load((sceneName + fileNameTwo))
    csv_three = pd.read_csv((sceneName + fileNameThree), header = None)

    frame_list = csv_three.iloc[0][:].unique()
    obs_peds = []
    
    for i in range(0,len(csv_two)):
        obs_peds.append(int(csv_two[i][0][0]))

    ped_num = int(max(csv.iloc[:][1])+1)
    print("a ", ped_num)
    
    ### Use numpy to populate 2D list of zeros, then convert to list so that variable lists can be stored
    phi_array = np.zeros((ped_num, ped_num))
    phi_array = phi_array.tolist()

    
    ped_range = list(range(0,ped_num))

    if int(sys.argv[1]) == 0:
        phi_type = "v"
    elif int(sys.argv[1]) == 1:
        phi_type = "g"
    elif int(sys.argv[1]) == 2:
        phi_type = "a"
    elif int(sys.argv[1]) == 3:
        phi_type = "h"
    elif int(sys.argv[1]) == 4:
        phi_type = "d"
    elif int(sys.argv[1]) == 5:
        phi_type = "c"

    elif int(sys.argv[1]) == 6:
        phi_type = "vga_05hdc"
    elif int(sys.argv[1]) == 7:
        phi_type = "ga_05hdc"
    elif int(sys.argv[1]) == 8:
        phi_type = "va_05hdc"
    elif int(sys.argv[1]) == 9:
        phi_type = "vg_05hdc"
    elif int(sys.argv[1]) == 10:
        phi_type = "vga_05dc"
    elif int(sys.argv[1]) == 11:
        phi_type = "vga_05hc"
    elif int(sys.argv[1]) == 12:
        phi_type = "vga_05hd"

    elif int(sys.argv[1]) == 13:
        phi_type = "a_05hdc"
    elif int(sys.argv[1]) == 14:
        phi_type = "v_05hdc"
    elif int(sys.argv[1]) == 15:
        phi_type = "va_05dc"
    elif int(sys.argv[1]) == 16:
        phi_type = "va_05hc"
    elif int(sys.argv[1]) == 17:
        phi_type = "va_05hd"

    elif int(sys.argv[1]) == 18:
        phi_type = "a_05hd"
    elif int(sys.argv[1]) == 19:
        phi_type = "v_05hd"
    elif int(sys.argv[1]) == 20:
        phi_type = "va_05d"
    elif int(sys.argv[1]) == 21:
        phi_type = "va_05h"

    elif int(sys.argv[1]) == 22:
        phi_type = "03v_hd"
    elif int(sys.argv[1]) == 23:
        phi_type = "03_hd"
    elif int(sys.argv[1]) == 24:
        phi_type = "03v_d"
    elif int(sys.argv[1]) == 25:
        phi_type = "03v_h"

    else:
        print("Incompatible input argument")

    print("phi_type: ", phi_type)

    print("CPU count: ", 8)
    pool = mp.Pool(10)
    tuples = []   
    for i in obs_peds:
        print("i is: " +str(i), flush = True)


        tuples = pool.starmap_async(preprocess_EDGE, [(i, j, phi_array[i][j], phi_array[j][i], csv, frame_list, phi_type) for j in ped_range]).get()
    

    
        for tup in tuples:
            if(tup[2] != [0]):
                if(phi_array[i][tup[1]]==0):
                    phi_array[i][tup[1]] = tup[2]

            if(tup[3] != [0]):
                if(phi_array[tup[1]][i]==0):
                    phi_array[tup[1]][i] = tup[3]

    pool.close()
    pool.join()

    #!# MI_Tensor doesnt use dtype=object, remove it if it causes an issue
    phi_array = np.asarray(phi_array, dtype = object)
    print("done")    

    np.save((sceneName + "phi/phi_tensor_{}.npy".format(str(phi_type))), phi_array)


####################################
####################################
if __name__ == "__main__":

    vers = "null"
    if sys.argv[2] == "raw":
        vers = "raw"
    elif sys.argv[2] == "noedge":
        vers = "noedge"
    elif sys.argv[2] == "corrected":
        vers = "corrected"
    else:
        print("incorrect version input")

    i=0
    ### Function options: 
    function = ""
    for root, subdir, file in os.walk("./" + vers):
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
                    
                elif f == ("pos_data.csv"):
                    fileNameThree = (f)
                    
            if fileName == "pos_data_interp.csv" and fileNameTwo == "obs.npy" and fileNameThree == ("pos_data.csv"):
                parallelPhi(directory, fileName, fileNameTwo, fileNameThree)
            else:
                print(directory, ": Doesn't contain necessary input files for phi calculation")
