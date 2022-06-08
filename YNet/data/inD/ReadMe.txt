The train-test-validation split used for inD was:

Train: 0-4, 7-13, 18-25, 30
Val: 5, 14-15, 26-27, 31
Test: 6, 16-17, 28-29, 32

The data is structures as:

- /
	- Train/
		- <Video>/
			- <Video>_recordingMeta.csv
			- <Video>_tracksMeta.csv
			- reference.png
	- Test/
		- same as above
	- Val/ 
		- same as above


The raw data isn't provided since it requires requested permission by the inD Dataset authors, as such we leave users of the code to request permission directly.