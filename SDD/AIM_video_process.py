#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 11:32:45 2020

@author: josh
"""


import cv2
import os

def convert(sceneName, inputFile):
    vidcap = cv2.VideoCapture(sceneName + inputFile)
    success, image = vidcap.read()
    print(vidcap.get(cv2.CAP_PROP_FPS))
    print(sceneName + inputFile)
    if success == True:
        print("success")
    count = 0
    ### This may throw an error when it passes the last frame, but saves all frames properly
    while success:
    	### While there is a next frame in the video, read it and save as a labeled jpg file
        try:
            success, image = vidcap.read()
            ### May need to ensure that the frames directory exists prior to running, or add a line to create it
            cv2.imwrite((sceneName + "frames/" + "frame%d.jpg" % count), image)
            if cv2.waitKey(10)==27:
                break
            count+=1
            print(sceneName)
        except:
            print("End of video")
            success = False
    return 0
    
i=0

### Loops over all subdirectories in ./data and converts any located video files into constituent frames
for root, subdir, file in os.walk("./data"):
  if len(file)>0:
      print(i)
      i+=1

      print(root)
      print(subdir)
      print(file)
      for f in file:
           if f == "video.mov":
               convert((root+"/"), f)