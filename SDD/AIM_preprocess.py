import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt




"""
To run: Prepare frames folder by running AIM_Video_Process.py
Ensure the Nx4 position csv file pos_data.csv is in ./data/SCENE_NAME/(corrected/raw)/VIDEO_NAME/

From there ADI_Preprocess.py will save the results of preprocessing the annotations and frames

Code adapted from SS-LSTM paper: SS-LSTM: A hierarchical LSTM model for pedestrian trajectory prediction

"""


class DataProcesser:

    ### Observed_Frame_Number is how many previous frames are used to calculate the predicted trajectory for the enxt Predicting_Frame_Number frames
    def __init__(self, data_dir, inputFile, observed_frame_num, predicting_frame_num):
        self.data_dir = data_dir
        self.inputFile = inputFile
        self.file_path = os.path.join(self.data_dir, inputFile)
        self.raw_data = None
        self.ped_num = None
        self.traj_data = []
        self.obs = []
        self.pred = []
        self.obs_length = observed_frame_num
        self.pred_length = predicting_frame_num

        ### Calls the necessary functions to set up
        self.from_csv()
        self.get_traj()
        self.get_obs_pred()

    def from_csv(self):
        print('Creating Raw Data from CSV file...')
        ### Because of the inclusion of label strings, numpy attempts to convert the strings to 
        ###    a different datatype, resulting in NaNs, this approach properly retains their data type
        data = pd.read_csv(self.file_path, header = None, delimiter = r',')
        data = np.asarray(data)
        data = data.transpose()
        self.raw_data = data
        ### the raw data reads the csv in by row, giving a 4xN matrix as a result. 
        ### This stores the number of unique identifiers in the 2nd row as the number of pedestrians
        self.ped_num = np.unique(self.raw_data[1, :])

    def get_traj(self):
        """
        reshape data format from [frame_ID, ped_ID, x-coord, y-coord]
        to pedestrian_num * [ped_ID, frame_ID, x-coord, y-coord]
        """

        for pedIndex in self.ped_num:
            traj = []
            for i in range(len(self.raw_data[1])):
                if self.raw_data[1][i] == pedIndex:
                    ### This appends in sequential order since the data is already ordered by frame number
                    traj.append([self.raw_data[1][i], self.raw_data[0][i], self.raw_data[2][i], self.raw_data[3][i], self.raw_data[4][i]])
            traj = np.reshape(traj, [-1, 5])
            self.traj_data.append(traj)
            ### Add the newly prepared trajectory to the object's existing list, given some filter function is true

        return self.traj_data

    def get_obs_pred(self):
        """
        get input observed data and output predicted data
        """
        count = 0
            
        ### Iterates over each agent's trajectory 
        ### traj_data is already in the order of ascending pedestrian number
        for pedIndex in range(len(self.traj_data)):
            
            ### This is to say if there is enough datapoints in the trajectory for this prediction to be made
                ### For example, in ETHhotel, only 117 agents have long enough trajectories to be included in obs and pred
            if len(self.traj_data[pedIndex]) >= 2 + self.obs_length + self.pred_length:
                obs_pedIndex = []
                pred_pedIndex = []
                count += 1
                ### Fetches all parts of the trajectory up to the inputs for obs and pred length
                for i in range(self.obs_length):
                    obs_pedIndex.append(self.traj_data[pedIndex][2 + i])
                for j in range(self.pred_length):
                    pred_pedIndex.append(self.traj_data[pedIndex][j + 2 + self.obs_length])

                obs_pedIndex = np.reshape(obs_pedIndex, [self.obs_length, 5])
                pred_pedIndex = np.reshape(pred_pedIndex, [self.pred_length, 5])

                self.obs.append(obs_pedIndex)
                self.pred.append(pred_pedIndex)

        self.obs = np.reshape(self.obs, [count, self.obs_length, 5])
        self.pred = np.reshape(self.pred, [count, self.pred_length, 5])

        return self.obs, self.pred




### image reading functions, not part of DataProcesser
def image_tensor(data_dir, frame_ID, image_width, image_height):
    img_dir = data_dir + "frame" + str(frame_ID) + '.jpg'
    img = cv2.imread(img_dir)
    img = cv2.resize(img, (image_width, image_height))

    return img

### Converts all frames in the directory into a single tensor of pixel values
def all_image_tensor(data_dir, obs, img_width, img_height):
    image = []

    for i in range(len(obs)):
        ### Gives the frame for the last observed position
        image.append(image_tensor(data_dir, int(obs[i][-1][1]), img_width, img_height))

    image = np.reshape(image, [len(obs), img_width, img_height, 3])

    return image



i=0

for root, subdir, file in os.walk("./annotations/"):
    if len(file)>0:
        print(i)
        i+=1
        
        print(root)
        print(subdir)

        for f in file:
            if f == "pos_data_temp.csv":
                directory = (root +"/")
                prep = DataProcesser(directory,f ,8,12)
                ### Save the obs and pred as npy files to be loaded during model script
                np.save((directory + 'obs'), prep.obs)
                np.save((directory + 'pred'), prep.pred)
