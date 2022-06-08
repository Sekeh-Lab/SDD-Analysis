import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import re





def visualize(csv, sceneName, scene, vers):
    csv = csv.transpose()
    if sceneName == "./coupa" + vers + "video0/" or sceneName == "./coupa" + vers + "video1/" or sceneName == "./coupa" + vers + "video2/" or sceneName == "./coupa" + vers + "video3/":
        range_x = 1980
        range_y = 1093
    
    elif sceneName == "./deathCircle" + vers + "video0/":
        range_x = 1630
        range_y = 1948
        
    elif sceneName == "./deathCircle" + vers + "video1/":
        range_x = 1409
        range_y = 1916
    
    elif sceneName == "./deathCircle" + vers + "video2/":
        range_x = 1436
        range_y = 1959

    elif sceneName == "./deathCircle" + vers + "video3/":
        range_x = 1400
        range_y = 1904
        
    elif sceneName == "./deathCircle" + vers + "video4/":
        range_x = 1452
        range_y = 1994
        
    elif sceneName == "./gates" + vers + "video0/" or sceneName == "./gates" + vers + "video2/":
        range_x = 1325
        range_y = 1973
        
    elif sceneName == "./gates" + vers + "video1/":
        range_x = 1425
        range_y = 1973
        
    elif sceneName == "./gates" + vers + "video3/":
        range_x = 1432
        range_y = 2002
        
    elif sceneName == "./gates" + vers + "video4/":
        range_x = 1434
        range_y = 1982
        
    elif sceneName == "./gates" + vers + "video5/":
        range_x = 1426
        range_y = 2011
    
    elif sceneName == "./gates" + vers + "video6/":
        range_x = 1326
        range_y = 2011
        
    elif sceneName == "./gates" + vers + "video7/" or sceneName == "./gates" + vers + "video8/":
        range_x = 1334
        range_y = 1982
    
    elif sceneName == "./hyang" + vers + "video0/":
        range_x = 1455
        range_y = 1925
        
    elif sceneName == "./hyang" + vers + "video1/":
        range_x = 1445
        range_y = 2002
        
    elif sceneName == "./hyang" + vers + "video2/":
        range_x = 1433
        range_y = 841

    elif sceneName == "./hyang" + vers + "video3/":
        range_x = 1433
        range_y = 741
        
    elif sceneName == "./hyang" + vers + "video4/":
        range_x = 1340
        range_y = 1730
        
    elif sceneName == "./hyang" + vers + "video5/":
        range_x = 1454
        range_y = 1991
        
    elif sceneName == "./hyang" + vers + "video6/":
        range_x = 1416
        range_y = 848     
        
    elif sceneName == "./hyang" + vers + "video7/":
        range_x = 1450
        range_y = 1940
        
    elif sceneName == "./hyang" + vers + "video8/" or sceneName == "./hyang" + vers + "video9/":
        range_x = 1350
        range_y = 1940
        
    elif sceneName == "./hyang" + vers + "video10/" or sceneName == "./hyang" + vers + "video11/":
        range_x = 1416
        range_y = 748
        
    elif sceneName == "./hyang" + vers + "video12/":
        range_x = 1316
        range_y = 848
        
    elif sceneName == "./hyang" + vers + "video13/" or sceneName == "./hyang" + vers + "video14/":
        range_x = 1316
        range_y = 748
        
    elif sceneName == "./little" + vers + "video0/":
        range_x = 1417
        range_y = 2019
        
    elif sceneName == "./little" + vers + "video1/" or sceneName == "./little" + vers + "video2/":
        range_x = 1322
        range_y = 1945
        
    elif sceneName == "./little" + vers + "video3/":
        range_x = 1422
        range_y = 1945
        
    elif sceneName == "./nexus" + vers + "video0/" or sceneName == "./nexus" + vers + "video2/":
        range_x = 1330
        range_y = 1947
        
    elif sceneName == "./nexus" + vers + "video1/":
        range_x = 1430
        range_y = 1947
        
    elif sceneName == "./nexus" + vers + "video3/" or sceneName == "./nexus" + vers + "video5/":
        range_x = 1184
        range_y = 1759
        
    elif sceneName == "./nexus" + vers + "video4/":
        range_x = 1284
        range_y = 1759
        
    elif sceneName == "./nexus" + vers + "video6/" or sceneName == "./nexus" + vers + "video8/":
        range_x = 1331
        range_y = 1962
        
    elif sceneName == "./nexus" + vers + "video7/":
        range_x = 1431
        range_y = 1962
        
    elif sceneName == "./nexus" + vers + "video9/":
        range_x = 1411
        range_y = 1980
        
        
    elif sceneName == "./nexus" + vers + "video10/" or sceneName == "./nexus" + vers + "video11/":
        range_x = 1311
        range_y = 1980
         
    elif sceneName == "./quad" + vers + "video0/" or sceneName == "./quad" + vers + "video1/" or sceneName == "./quad" + vers + "video2/" or sceneName == "./quad" + vers + "video3/":
        range_x = 1983
        range_y = 1088
                
    elif sceneName == "./bookstore" + vers + "video0/":
        range_x = 1424
        range_y = 1088
    
    elif sceneName == "./bookstore" + vers + "video1/" or sceneName == "./bookstore" + vers + "video2/":
        range_x = 1422
        range_y = 1079

    elif sceneName == "./bookstore" + vers + "video3/" or sceneName == "./bookstore" + vers + "video4/" or sceneName == "./bookstore" + vers + "video5/" or sceneName == "./bookstore" + vers + "video6/":
        range_x = 1322
        range_y = 1079
        
    else:
        print(sceneName, ": Dataset not implemented yet!")

	### Get array of all pedestrian IDs        
    peds = np.unique(csv.iloc[:,1])


    for ind,i in enumerate(peds):
        label_i = ""
        i = int(i)
        traj_data_i = csv[csv.iloc[:,1].astype(int) == i]
        traj_data_i.index = range(traj_data_i.shape[0])
   
        frames = traj_data_i.iloc[:,0]

        ### Select a marker based on vehicle vs foot-traffic
        if traj_data_i.iloc[0][4] == "Pedestrian":
            label_i = "Circle"
        elif traj_data_i.iloc[0][4] == "Skater":
            label_i = "Circle"
        elif traj_data_i.iloc[0][4] == "Biker":
            label_i = "Circle"
        elif traj_data_i.iloc[0][4] == "Cart":
            label_i = "Square"
        elif traj_data_i.iloc[0][4] == "Bus":
            label_i = "Square"
        elif traj_data_i.iloc[0][4] == "Car":
            label_i = "Square"
        else:
            print("Unrecognized Agent Class for: ", traj_data_i.iloc[0][0])
        
            
        if not os.path.exists("./agents/" + sceneName[2:]):
            os.makedirs("./agents/" + sceneName[2:])

        ### Get current video name from the os.walk root path
        valid = re.compile(r"^./(\S+)/(\S+)/(\S+)/")
        if(valid.match(sceneName)):
            matchText = valid.match(sceneName)
            vid = matchText.group(3)
            
            
        ### Use the first frame jpg of the pedestrians trajectory to draw markers on    
        img = cv2.imread(scene + "/frames/" + vid + "/frames/frame" + str(traj_data_i.iloc[0][0]) + '.jpg')


        ### Make the markers shrink over the course of the trajectory to indicate direction of travel
        for index, frame in enumerate(traj_data_i.iloc[:,0]):
            if index == 0:
                length = 10
            elif index <= 5:
                length = 8
            if index <= 10:
                length = 6
            else:
                length = 4
                
                
            x = int(float(traj_data_i.iloc[index,2])*range_x)
            y = int(float(traj_data_i.iloc[index,3])*range_y)
          
            if label_i == "Circle":
                if index <= 8:
                    img = cv2.circle(img,(x,y), length, (0,0,255), -1)
                elif index > 8 and index <= 20:
                    img = cv2.circle(img,(x,y), length, (0,255,0), -1)
                elif index >20:
                    img = cv2.circle(img,(x,y), length, (255,0,0), -1)
            elif label_i == "Square":
                if index <= 8:
                    img = cv2.rectangle(img,(x-int(length/2),y-int(length/2)), (x+int(length/2),y+int(length/2)), (0,0,255), -1)
                elif index > 8 and index <= 20:
                    img = cv2.rectangle(img,(x-int(length/2),y-int(length/2)), (x+int(length/2),y+int(length/2)), (0,255,0), -1)
                elif index >20:
                    img = cv2.rectangle(img,(x-int(length/2),y-int(length/2)), (x+int(length/2),y+int(length/2)), (255,0,0), -1)
            else:
                print("Invalid input shape for agent i marking!")


        cv2.imwrite("./agents/" + sceneName[2:] + "agent_{}.jpg".format(str(i)), img)







print("started")

### Change as needed
scene = "./quad"
vers = "/corrected/"
# vers = "/raw/"

i = 0
for root, subdir, file in os.walk(scene + vers):
    if len(file)>0:
        # print(i)
        i+=1
        
        # print(root)
        # print(subdir)
        #print(file)
        for f in file:
            if f == "pos_data.csv":
                sceneName = (root +"/")
                print(sceneName)
                print(sceneName[2:])
                csv = pd.read_csv((sceneName + "pos_data.csv"), header = None)
                visualize(csv, sceneName, scene, vers)

    























