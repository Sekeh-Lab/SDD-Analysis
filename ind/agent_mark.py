import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import re





def visualize(fileOne, sceneName, background):
    csv = pd.read_csv((sceneName + fileOne), header = None)
    csv = csv.transpose()
    
    imgplt = plt.imread((sceneName + background))
    width = imgplt.shape[1]
    height = imgplt.shape[0]
    


    peds = np.unique(csv.iloc[:,1])

    for ind,i in enumerate(peds):
        label_i = ""
        i = int(i)
        traj_data_i = csv[csv.iloc[:,1].astype(int) == i]
        traj_data_i.index = range(traj_data_i.shape[0])
   
        frames = traj_data_i.iloc[:,0]


        if traj_data_i.iloc[0][4] == "pedestrian":
            label_i = "Circle"
        elif traj_data_i.iloc[0][4] == "bicycle":
            label_i = "Circle"
        else:
            label_i = "Square"
        
        ### Something like this to get all frames in the shared trajectory
            


        if not os.path.exists("./agents/" + sceneName[2:]):
            os.makedirs("./agents/" + sceneName[2:])
            
        img = cv2.imread((sceneName + background))


        for index, frame in enumerate(traj_data_i.iloc[:,0]):

            if index == 0:
                length = 10
            elif index <= 5:
                length = 8
            if index <= 10:
                length = 6
            else:
                length = 4
                
                
            x = int(float(traj_data_i.iloc[index,2])*width)
            y = int(float(traj_data_i.iloc[index,3])*height)
           

            if label_i == "Circle":
                img = cv2.circle(img,(x,y), length, (255,0,0), -1)
            elif label_i == "Square":
                img = cv2.rectangle(img,(x-int(length/2),y-int(length/2)), (x+int(length/2),y+int(length/2)), (255,0,0), -1)
            else:
                print("Invalid input shape for agent i marking!")



        cv2.imwrite("./agents/" + sceneName[2:] + "agent_{}.jpg".format(str(i)), img)







print("started")




i = 0
for root, subdir, file in os.walk("./"):
    if len(file)>0:
        # print(i)
        i+=1
        
        # print(root)
        # print(subdir)
        #print(file)
        fileOne = ""
        fileTwo = ""
        
        sceneName = (root +"/")
        for f in file:
            if f == "pos_data.csv":
                fileOne = "pos_data.csv"
            if f == "background.png":
                fileTwo = "background.png"

        if fileOne == "pos_data.csv" and fileTwo == "background.png":
            print(sceneName)
            print(sceneName[2:])
            visualize(fileOne, sceneName, fileTwo)
        else:
            print("Doesnt contain correct files")
    


























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

