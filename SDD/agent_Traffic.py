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
        
    peds = np.unique(csv.iloc[:,1])
    # peds = np.array([10, 49, 87, 97])
    # traj_data_temp = csv[csv.iloc[:,1].astype(int) == 97]

    # img = cv2.imread(scene + "/frames/frame" + str(traj_data_temp.iloc[0][0]) + '.jpg')
    valid = re.compile(r"^./(\S+)/(\S+)/(\S+)/")
    if(valid.match(sceneName)):
        matchText = valid.match(sceneName)
        vid = matchText.group(3)

    img0 = cv2.imread(scene + "/frames/" + vid + "/frames/frame400.jpg")
    img1 = cv2.imread(scene + "/frames/" + vid + "/frames/frame400.jpg")
    img2 = cv2.imread(scene + "/frames/" + vid + "/frames/frame400.jpg")

    for ind,i in enumerate(peds):
        label_i = ""
        i = int(i)
        traj_data_i = csv[csv.iloc[:,1].astype(int) == i]
        traj_data_i.index = range(traj_data_i.shape[0])
   
        frames = traj_data_i.iloc[:,0]


        if traj_data_i.iloc[0][4] == "Pedestrian":
            label_i = "Ped"
        elif traj_data_i.iloc[0][4] == "Skater":
            label_i = "FastPed"
        elif traj_data_i.iloc[0][4] == "Biker":
            label_i = "FastPed"
        elif traj_data_i.iloc[0][4] == "Cart":
            label_i = "Vehicle"
        elif traj_data_i.iloc[0][4] == "Bus":
            label_i = "Vehicle"
        elif traj_data_i.iloc[0][4] == "Car":
            label_i = "Vehicle"
        else:
            print("Unrecognized Agent Class for: ", traj_data_i.iloc[0][0])
        
        ### Something like this to get all frames in the shared trajectory
            


        if not os.path.exists("./traffic/" + sceneName[2:]):
            os.makedirs("./traffic/" + sceneName[2:])


            
            
            


        for index, frame in enumerate(traj_data_i.iloc[:,0]):

            length = 4
                
                
            x = int(float(traj_data_i.iloc[index,2])*range_x)
            y = int(float(traj_data_i.iloc[index,3])*range_y)
            # if ind == 0:
            #     if label_i == "Circle":
            #         img = cv2.circle(img,(x,y), length, (0,0,255), -1)
            #     elif label_i == "Square":
            #         img = cv2.rectangle(img,(x-int(length/2),y-int(length/2)), (x+int(length/2),y+int(length/2)), (255,0,0), -1)
            #     else:
            #         print("Invalid input shape for agent i marking!")
            # elif ind == 1:
            #     if label_i == "Circle":
            #         img = cv2.circle(img,(x,y), length, (0,255,0), -1)
            #     elif label_i == "Square":
            #         img = cv2.rectangle(img,(x-int(length/2),y-int(length/2)), (x+int(length/2),y+int(length/2)), (255,0,0), -1)
            #     else:
            #         print("Invalid input shape for agent i marking!")
            # elif ind == 2:
            #     if label_i == "Circle":
            #         img = cv2.circle(img,(x,y), length, (255,0,255), -1)
            #     elif label_i == "Square":
            #         img = cv2.rectangle(img,(x-int(length/2),y-int(length/2)), (x+int(length/2),y+int(length/2)), (255,0,0), -1)
            #     else:
            #         print("Invalid input shape for agent i marking!")
            # else:
            if label_i == "Ped":
                img0 = cv2.circle(img0,(x,y), length, (0,0,255), -1)
            elif label_i == "FastPed":
                img0 = cv2.circle(img0,(x,y), length, (100,255,255), -1)
            elif label_i == "Vehicle":
                img0 = cv2.circle(img0,(x,y), length, (255,0,0), -1)
            else:
                print("Invalid input shape for agent i marking!")

                        # cv2.imwrite("./agents/" + sceneName[2:] + "agent_{}.jpg".format(str(i)), img)

    # cv2.imwrite("./traffic/" + sceneName[2:] + "agent_Traffic_ped.jpg", img0, dpi = 300)
    # cv2.imwrite("./traffic/" + sceneName[2:] + "agent_Traffic_bike.jpg", img1, dpi = 300)
    # cv2.imwrite("./traffic/" + sceneName[2:] + "agent_Traffic_veh.jpg", img2, dpi = 300)
    cv2.imwrite("./traffic/" + sceneName[2:] + "agent_Traffic.jpg", img0)







print("started")


# scene = "./"
# if int(sys.argv[1])==0:
#     scene = "./bookstore"
# elif int(sys.argv[1])==1:
#     scene = "./coupa"
# elif int(sys.argv[1])==2:
#     scene = "./deathCircle"
# elif int(sys.argv[1])==3:
#     scene = "./gates"
# elif int(sys.argv[1])==4:
#     scene = "./hyang"
# elif int(sys.argv[1])==5:
#     scene = "./little"
# elif int(sys.argv[1])==6:
#     scene = "./nexus"
# elif int(sys.argv[1])==7:
#     scene = "./quad"
# else:
#     print("Incorrect scene input")

scene = "./quad"
vers = "/corrected/"

# vers = "/"
# if sys.argv[2] == "raw":
#     vers = "/raw/"
# elif sys.argv[2] == "noedge":
#     vers = "/noedge/"
# elif sys.argv[2] == "corrected":
#     vers = "/corrected/"
# else:
#     print("Incorrect version input")

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

    


























# ### image reading functions, not part of DataProcesser
# def image_tensor(data_dir, frame_ID, new_size):
#     img_dir = data_dir + "frame" + str(frame_ID) + '.jpg'
#     img = cv2.imread(img_dir)
    
#     old_size = img.shape[:2] # old_size is in (height, width) format
    
#     ratio = float(new_size)/max(old_size)
#     scaled_size = tuple([int(x*ratio) for x in old_size])
    
#     # new_size should be in (width, height) format
    
#     img_scaled = cv2.resize(img, (scaled_size[1], scaled_size[0]))
    
#     delta_w = new_size - scaled_size[1]
#     delta_h = new_size - scaled_size[0]
#     top, bottom = delta_h//2, delta_h-(delta_h//2)
#     left, right = delta_w//2, delta_w-(delta_w//2)
    
#     color = [0, 0, 0]
#     new_img = cv2.copyMakeBorder(img_scaled, top, bottom, left, right, cv2.BORDER_CONSTANT,
#         value=color)
    
#     # plt.imshow(im, gra)
#     # plt.show()
#     return new_img

# ### Converts all frames in the directory into a single tensor of pixel values
# ### Im almost certain the author had the height and width backwards, but had them flipped during reshaping to negate the issue, so I undid that
#     ### The end result should be the same, just less confusing, the height is 576, the width is 720
# def all_image_tensor(data_dir, obs, new_size):

#     image = []

#     for i in range(len(obs)):
#         ### Gives the frame for the last observed position
#         image.append(image_tensor(data_dir, int(obs[i][-1][1]), new_size))

#     image = np.reshape(image, [len(obs), new_size, new_size, 3])

#     return image


# ### Save the obs and pred as npy files to be loaded during model script

# i=0

# for root, subdir, file in os.walk("./coupa/video1"):
#     if len(file)>0:
#         print(i)
#         i+=1
        
#         print(root)
#         print(subdir)
#         #print(file)
#         for f in file:
#             if f == "pos_data_temp.csv":
#                 directory = (root +"/")
#                 obs = np.load(directory+"obs.npy")
# #                prep = DataProcesser(directory,f ,8,12)
# #                np.save((directory + 'obs'), prep.obs)
# #                np.save((directory + 'pred'), prep.pred)
                
#                 #image_array = all_image_tensor((directory + 'frames/'), prep.obs, 720, 576)
#                 image_array_large = all_image_tensor((directory + 'frames/'), obs, 720)
#                 #np.save((directory + 'img_data'), image_array)
#                 np.save((directory + 'img_data_fixed'), image_array_large)


# ### Change directory as needed

