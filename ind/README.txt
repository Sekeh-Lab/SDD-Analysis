The accompanying code is used to preprocess the inD Dataset into the same format as the processed SDD data for comparison and analysis. At the time of publication, the inD Dataset requires users to request access from its authors, as such in order to respect this decision we omit inclusion of the raw data within this repository. It can be found at https://www.ind-dataset.com/. Once downloaded, the data should encompass 33 annotations across 4 intersections. We restructure this by intersection number (1-4) as follows:

	- ./data
		- <Scene>
			- <Video>

Unlike with the SDD structure, since there are no lost coordinates we don't maintain a "corrected" and "raw" version for analysis. As with the SDD, we only include one of the 33 annotations to serve as an example of the expected results while reducing repository size.


To process the data, the following scripts must be run in order:
1. Text_Preprocess_One: Initial step in processing the inD annotations. Outputs the same format as in the SDD processing, after this step the files are identical to the processing files used for the SDD.
2. Text_Preprocess_Two: Run once to perform the next processing stage for both the corrected and raw annotation data.
3. Text_Preprocess_Three: Run once as in the previous step. Outputs the files used for analysis.
4. AIM_Preprocess: Splits the annotations in files output by step 4 into annotations of the observed 8 time points at the start of each trajectory (obs.npy) and the 12 predicted time points (pred.npy)

EDGE_MI.py and EDGE_Phi.py can be run to calculate Mutual Information and phi values for both the corrected and raw annotation data. Lastly Phi_MI_Summation_Decay.py can be
 used to calculate the AIM as was done for the paper. 

At this point:
- Agent_Mark.py can be used produce figures showing each individual's trajectory as shown in the provided folder "Agent_Trajectories" 
- Agent_Traffic.py can be used to produce figures showing all trajectories within each recording
- Visualization.py can be used to produce plots of various metrics (phi, MI, AIM) over a pairwise interaction between two agents


 