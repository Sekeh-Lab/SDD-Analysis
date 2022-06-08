Prior to processing the data, the annotation files from the SDD should be organized in the following path structure:
	- Data
		- <Scene>
			- raw
				- <Videos>
			- corrected
				- <Videos>
			- frames
				- <Videos>
					- Video File
Where <Scene> represents one of the the 8 scenes' directories and <Videos> is the directory of one of the videos within that scene (not the video file itself). This
 structure can be seen in the provided Quad directory.
 
To process the data, the following scripts must be run in order:
1. AIM_Video_Process: For each video file, splits it into constituent frames and saves them in the same directory as the video, after which the videos can be moved/deleted.
2. Text_Preprocess_One: Run with vers="raw" again with vers="corrected" (hard-coded into the os.walk loop at the end of the file). Initial step in processing the SDD annotations.
3. Text_Preprocess_Two: Run once to perform the next processing stage for both the corrected and raw annotation data.
4. Text_Preprocess_Three: Run once as in the previous step. Outputs the files used for analysis.
5. AIM_Preprocess: Splits the annotations in files output by step 4 into annotations of the observed 8 time points at the start of each trajectory (obs.npy) and the 12 predicted time points (pred.npy)

Within "./data/<Scene>" EDGE_MI.py and EDGE_Phi.py can be run to calculate Mutual Information and phi values for both the corrected and raw annotation data. Lastly Phi_MI_Summation_Decay.py can be
 used to calculate the AIM as was done for the paper. 

At this point:
- Agent_Mark.py can be used produce figures showing each individual's trajectory. 
- Lost_Occurences.py can be used to output a csv file of the number of overall annotations in each scene labeled as "lost", as well as how many occur at the beginning, end, and middle of trajectories.
- Agent_Traffic.py can plot all trajectories in each given video.
- Visualizations.py can be used to produce plots of the interaction between a given pair of agents in a video using various metrics such as AIM, velocity, etc.


 