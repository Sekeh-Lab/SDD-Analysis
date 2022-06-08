import os
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import time
import sys
import re
from matplotlib.lines import Line2D

mpl.use("Agg")

"""
Needed:
- pull the paired trajectory in question
    - pull the full paired trajectories, theres no need to use obs and pred here

- pull its associated ADIs

- pull the first frame of the paired trajectory
- based on the type of each agent, color code their paths consistently
- Make an image for each frame in the trajectory pair such that the people move along their paths



Paired Trajectories:
- Can just copy the code from the Phi script to produce the joint trajectory
- 
"""





def visualize(peds, csv, frame_list, measure, measureVals, sceneName, title, case, vers, scene):
    # csv = csv.transpose()
    maxVal = 0
    cmap = cm.get_cmap('gnuplot2')
    
    
    for i in range(0, len(measureVals[0])):
        for j in range(0, len(measureVals[0])):
            if measureVals[i][j] != 0:
                for frame in measureVals[i][j]:
                    if frame[1] > maxVal:
                        maxVal = frame[1]

    
    if sceneName == "./coupa/" + vers + "video0/" or sceneName == "./coupa/" + vers + "video1/" or sceneName == "./coupa/" + vers + "video2/" or sceneName == "./coupa/" + vers + "video3/":
        range_x = 1980
        range_y = 1093
    
    elif sceneName == "./deathCircle/" + vers + "video0/":
        range_x = 1630
        range_y = 1948
        
    elif sceneName == "./deathCircle/" + vers + "video1/":
        range_x = 1409
        range_y = 1916
    
    elif sceneName == "./deathCircle/" + vers + "video2/":
        range_x = 1436
        range_y = 1959

    elif sceneName == "./deathCircle/" + vers + "video3/":
        range_x = 1400
        range_y = 1904
        
    elif sceneName == "./deathCircle/" + vers + "video4/":
        range_x = 1452
        range_y = 1994
        
    elif sceneName == "./gates/" + vers + "video0/" or sceneName == "./gates/" + vers + "video2/":
        range_x = 1325
        range_y = 1973
        
    elif sceneName == "./gates/" + vers + "video1/":
        range_x = 1425
        range_y = 1973
        
    elif sceneName == "./gates/" + vers + "video3/":
        range_x = 1432
        range_y = 2002
        
    elif sceneName == "./gates/" + vers + "video4/":
        range_x = 1434
        range_y = 1982
        
    elif sceneName == "./gates/" + vers + "video5/":
        range_x = 1426
        range_y = 2011
    
    elif sceneName == "./gates/" + vers + "video6/":
        range_x = 1326
        range_y = 2011
        
    elif sceneName == "./gates/" + vers + "video7/" or sceneName == "./gates/" + vers + "video8/":
        range_x = 1334
        range_y = 1982
    
    elif sceneName == "./hyang/" + vers + "video0/":
        range_x = 1455
        range_y = 1925
        
    elif sceneName == "./hyang/" + vers + "video1/":
        range_x = 1445
        range_y = 2002
        
    elif sceneName == "./hyang/" + vers + "video2/":
        range_x = 1433
        range_y = 841

    elif sceneName == "./hyang/" + vers + "video3/":
        range_x = 1433
        range_y = 741
        
    elif sceneName == "./hyang/" + vers + "video4/":
        range_x = 1340
        range_y = 1730
        
    elif sceneName == "./hyang/" + vers + "video5/":
        range_x = 1454
        range_y = 1991
        
    elif sceneName == "./hyang/" + vers + "video6/":
        range_x = 1416
        range_y = 848     
        
    elif sceneName == "./hyang/" + vers + "video7/":
        range_x = 1450
        range_y = 1940
        
    elif sceneName == "./hyang/" + vers + "video8/" or sceneName == "./hyang/" + vers + "video9/":
        range_x = 1350
        range_y = 1940
        
    elif sceneName == "./hyang/" + vers + "video10/" or sceneName == "./hyang/" + vers + "video11/":
        range_x = 1416
        range_y = 748
        
    elif sceneName == "./hyang/" + vers + "video12/":
        range_x = 1316
        range_y = 848
        
    elif sceneName == "./hyang/" + vers + "video13/" or sceneName == "./hyang/" + vers + "video14/":
        range_x = 1316
        range_y = 748
        
    elif sceneName == "./little/" + vers + "video0/":
        range_x = 1417
        range_y = 2019
        
    elif sceneName == "./little/" + vers + "video1/" or sceneName == "./little/" + vers + "video2/":
        range_x = 1322
        range_y = 1945
        
    elif sceneName == "./little/" + vers + "video3/":
        range_x = 1422
        range_y = 1945
        
    elif sceneName == "./nexus/" + vers + "video0/" or sceneName == "./nexus/" + vers + "video2/":
        range_x = 1330
        range_y = 1947
        
    elif sceneName == "./nexus/" + vers + "video1/":
        range_x = 1430
        range_y = 1947
        
    elif sceneName == "./nexus/" + vers + "video3/" or sceneName == "./nexus/" + vers + "video5/":
        range_x = 1184
        range_y = 1759
        
    elif sceneName == "./nexus/" + vers + "video4/":
        range_x = 1284
        range_y = 1759
        
    elif sceneName == "./nexus/" + vers + "video6/" or sceneName == "./nexus/" + vers + "video8/":
        range_x = 1331
        range_y = 1962
        
    elif sceneName == "./nexus/" + vers + "video7/":
        range_x = 1431
        range_y = 1962
        
    elif sceneName == "./nexus/" + vers + "video9/":
        range_x = 1411
        range_y = 1980
        
        
    elif sceneName == "./nexus/" + vers + "video10/" or sceneName == "./nexus/" + vers + "video11/":
        range_x = 1311
        range_y = 1980
  
          
    elif sceneName == "./quad/" + vers + "video0/" or sceneName == "./quad/" + vers + "video1/" or sceneName == "./quad/" + vers + "video2/" or sceneName == "./quad/" + vers + "video3/":
        range_x = 1983
        range_y = 1088
            
        
    elif sceneName == "./bookstore/" + vers + "video0/":
        range_x = 1424
        range_y = 1088
    

    elif sceneName == "./bookstore/" + vers + "video1/" or sceneName == "./bookstore/" + vers + "video2/":
        range_x = 1422
        range_y = 1079

    elif sceneName == "./bookstore/" + vers + "video3/" or sceneName == "./bookstore/" + vers + "video4/" or sceneName == "./bookstore/" + vers + "video5/" or sceneName == "./bookstore/" + vers + "video6/":
        range_x = 1322
        range_y = 1079
        
    else:
        print(sceneName, ": Dataset not implemented yet!")
    



    valid = re.compile(r"^./(\S+)/(\S+)/(\S+)/")
    if(valid.match(sceneName)):
        matchText = valid.match(sceneName)
        vid = matchText.group(3)
        
        
        




    for ped in peds:
    
        i = ped[0]
        j = ped[1]
    
        if (j != i):
            if not os.path.exists("./interactions/" +  sceneName[2:] + "case_{}/{}/".format(case, measure) + "test_i{}_j{}_{}".format(str(i), str(j), str(title))):
                os.makedirs("./interactions/" + sceneName[2:] + "case_{}/{}/".format(case, measure) + "test_i{}_j{}_{}".format(str(i), str(j), str(title)))

            vals_i = measureVals[i,j]
            vals_j = measureVals[j,i]
    


            label_i = ""
            label_j = ""
            traj_data_i = csv[csv[1] == int(i)]
            traj_data_i.index = range(traj_data_i.shape[0])
            
            ### Extract rows belonging to agents i and j, then reset the row names.
            traj_data_j = csv[csv[1] == int(j)]
            traj_data_j.index = range(traj_data_j.shape[0])
            
            
           
            ### Prune rows belonging to frames that don't contain both agents
            traj_shared_i = traj_data_i[traj_data_i.iloc[:][0].isin(traj_data_j.iloc[:][0])]
            traj_shared_j = traj_data_j[traj_data_j.iloc[:][0].isin(traj_data_i.iloc[:][0])]
            ### Remove the frame and ped_ID columns, storing only the coordinate data
            frames = traj_shared_i[[0]]
            # traj_shared_i = traj_shared_i[[2,3]]
            # traj_shared_j = traj_shared_j[[2,3]]
            ### Store the pair of trajectory data for agents i and j
            traj_data = np.asarray([traj_shared_i.to_numpy(), traj_shared_j.to_numpy()])
            

            # plotvals_i = vals_i[vals_i[:][0] >= int(traj_shared_i.iloc[0,0]) and vals_i[:][0] <= int(traj_shared_i.iloc[-1,0])] 
            # plotvals_j = vals_j[vals_j[:][0] >= int(traj_shared_j.iloc[0,0]) and vals_j[:][0] <= int(traj_shared_j.iloc[-1,0])] 
            plotvals_i = pd.DataFrame.from_records(vals_i, columns = ["frame", "measure"])
            plotvals_j = pd.DataFrame.from_records(vals_j, columns = ["frame", "measure"])
            ### At this point traj_data contains the shared trajectory to be visualized
    
            # plotvals_i = plotvals_i[(plotvals_i.iloc[:,0].astype(int)) >= int(traj_shared_i.iloc[0,0]) and (plotvals_i.iloc[:,0].astype(int)) <= int(traj_shared_i.iloc[-1,0])]
            # plotvals_j = plotvals_j[(plotvals_j.iloc[:,0].astype(int)) >= int(traj_shared_j.iloc[0,0]) and (plotvals_j.iloc[:,0].astype(int)) <= int(traj_shared_j.iloc[-1,0])]

            plotvals_i = plotvals_i[(plotvals_i.iloc[:,0].astype(int)).between(int(traj_shared_i.iloc[0,0]), int(traj_shared_i.iloc[-1,0]))]
            plotvals_j = plotvals_j[(plotvals_j.iloc[:,0].astype(int)).between(int(traj_shared_j.iloc[0,0]), int(traj_shared_j.iloc[-1,0]))]

            label_i = "o"
            label_j = "^"
            
            ### Something like this to get all frames in the shared trajectory
            for frame, frame_outer in enumerate(traj_data[0][-1:,0]):

                file = scene + "frames/" + vid + "/frames/frame" + str(frame_outer) + '.jpg'
                # file = scene + "frames/frame" + str(frame_outer) + '.jpg'
                # img = cv2.imread(scene + "frames/" + vid + "/frames/frame" + str(traj_data_i.iloc[0][0]) + '.jpg')
                img = plt.imread(file, format = 'jpg')
                plt.imshow(img)
                # plt.savefig("./interactions/" + sceneName[2:] + "test_i{}_j{}_frame{}.png".format(str(i), str(j), str(frame_outer)))

                val_i = 0
                val_j = 0

                for index, frame_inner in enumerate(traj_data[0][:,0]):

                    frame_inner = int(frame_inner)
                    if measure == "phi":
                        for phi in vals_i:
                            if int(phi[0]) == frame_inner:
                                val_i = phi[1]
                        for phi in vals_j:
                            if int(phi[0]) == frame_inner:
                                val_j = phi[1]
                                
                    elif measure == "MI":
                        for MI in vals_i:
                            if int(MI[0]) == frame_inner:
                                val_i = MI[1]
                        for MI in vals_j:
                            if int(MI[0]) == frame_inner:
                                val_j = MI[1]
                                
                    elif measure == "ADI":
                        for ADI in vals_i:
                            if int(ADI[0]) == frame_inner:
                                val_i = ADI[1]
                        for ADI in vals_j:
                            if int(ADI[0]) == frame_inner:
                                val_j = ADI[1]


                    if int(frame_inner) == int(frame_outer):
                        length = 80
                    else:
                        length = 5
    
                    x_i = float(traj_data[0][index,2])*range_x
                    y_i = float(traj_data[0][index,3])*range_y


                    x_j = float(traj_data[1][index,2])*range_x
                    y_j = float(traj_data[1][index,3])*range_y

                    c_i = [cmap(val_i/maxVal)]
                    c_j = [cmap(val_j/maxVal)]


                    plt.scatter(x_i, y_i, marker = label_i, s = length, c = c_i)
                    plt.scatter(x_j, y_j, marker = label_j, s = length, c = c_j)

                # plt.title("Agents i:{} & j:{} for measure: {}".format(str(i), str(j), str(title[10:])))
                legend_elements = [Line2D([0], [0], marker='o', linestyle = "None", color='black', label='Agent {}'.format(str(i)), markerfacecolor='black', markersize=8),
                           Line2D([0], [0], marker='^', linestyle = "None", color='black', label='Agent {}'.format(str(j)), markerfacecolor='black', markersize=8)]

                # Create the figure
                # plt.legend(handles=legend_elements,prop = {"size":10, "weight":"bold"}, loc = (.025,0.85))
                plt.legend(handles=legend_elements,prop = {"size":10, "weight":"bold"}, loc = "lower right")
                plt.savefig("./interactions/" + sceneName[2:] + "case_{}/{}/".format(case, measure) + "test_i{}_j{}_{}/".format(str(i), str(j), str(title)) + "test_i{}_j{}_frame{}.jpg".format(str(i), str(j), str(frame_outer)), dpi = 200, bbox_inches = "tight")
                plt.clf()

                plt.colorbar(mpl.cm.ScalarMappable(norm = mpl.colors.Normalize(vmin=0, vmax = maxVal), cmap = "gnuplot2"), ticks = np.linspace(0, maxVal, 10, endpoint=True))
                plt.savefig("./interactions/" + sceneName[2:] + "case_{}/{}/".format(case, measure) + "test_i{}_j{}_{}/".format(str(i), str(j), str(title)) + "bar_i{}_j{}_frame{}.jpg".format(str(i), str(j), str(frame_outer)), dpi = 200, bbox_inches = "tight")
                plt.clf()
                # ax[1].plot(*zip(*val_j),c = "green")
                # plt.clim(0, maxMI)
                # plt.colorbar(ticks = np.linspace(0, maxMI, 10, endpoint=True), c = get_cmap("ocean"))
                plt.figure(figsize = (5,5), linewidth = 10)
                ax = plt.gca()
                ax.set_aspect(75)
                plt.rc('ytick',labelsize = 12)
                plt.rc('xtick',labelsize = 12)
                plt.locator_params(axis="x", nbins=6)
                plt.locator_params(axis="y", nbins=6)
                plt.plot(plotvals_i.iloc[:,0], plotvals_i.iloc[:,1], c = "red", linewidth = 3, label = "Agent {}".format(str(i)))
                plt.plot(plotvals_i.iloc[:,0], plotvals_j.iloc[:,1], c = "blue", linewidth = 3, linestyle = "dashed", label = "Agent {}".format(str(j)))

                plt.xlabel("Frame Number (30 fps)", fontsize = 13, fontweight = "bold")
                plt.ylabel("Rho Value", fontsize = 13, fontweight = "bold")
                # plt.ylabel("{} Value".format(str(measure)), fontsize = 13, fontweight = "bold")

                plt.legend(prop = {"size":12, "weight":"bold"})
                plt.savefig("./interactions/" + sceneName[2:] + "case_{}/{}/".format(case, measure) + "test_i{}_j{}_{}/".format(str(i), str(j), str(title)) + "graph_i{}_j{}_frame{}.jpg".format(str(i), str(j), str(frame_outer)), dpi = 300, bbox_inches = "tight")

                plt.clf()

                plt.plot(plotvals_i.iloc[:,0], plotvals_i.iloc[:,1], c = "red", linewidth = 3, label = "Agent {}".format(str(i)))
                legend_elements = [Line2D([0], [0], marker='o', linestyle = "None", color=(0,0,1), label='Agent {}'.format(str(i)), markerfacecolor=(0,0,1), markersize=8),
                           Line2D([0], [0], marker='o', linestyle = "None", color=(0.4,1,1), label='Agent {}'.format(str(j)), markerfacecolor=(0.4,1,1), markersize=8),
                           Line2D([0], [0], marker='o', linestyle = "None", color=(1,0,0), label='Agent {}'.format(str(j)), markerfacecolor=(1,0,0), markersize=8)]

                # Create the figure
                # plt.legend(handles=legend_elements,prop = {"size":10, "weight":"bold"}, loc = (.025,0.85))
                plt.legend(handles=legend_elements,prop = {"size":10, "weight":"bold"}, loc = "upper right")
                plt.savefig("./interactions/" + sceneName[2:] + "case_{}/{}/".format(case, measure) + "test_i{}_j{}_{}/".format(str(i), str(j), str(title)) + "legend_i{}_j{}_frame{}.jpg".format(str(i), str(j), str(frame_outer)), dpi = 300, bbox_inches = "tight")



print("started")

peds = []
case = 100
sceneName = ""
scene = ""

vers = ""
if sys.argv[2] == "raw":
    vers = "raw/"
elif sys.argv[2] == "noedge":
    vers = "noedge/"
elif sys.argv[2] == "corrected":
    vers = "corrected/"
else:
    print("incorrect version input")




if int(sys.argv[1]) == 0:
    scene = "./bookstore/"
    sceneName = scene + vers +"video0/"
    case = 0
    peds = [[5, 25]]

elif int(sys.argv[1]) == 1:
    scene = "./bookstore/"
    sceneName = scene + vers + "video0/"
    case =  1
    peds = [[53, 54], [53, 55]]

elif int(sys.argv[1]) == 2:
    scene = "./bookstore/"
    sceneName = scene + vers + "video0/"
    case =  2
    peds = [[73, 212]]

elif int(sys.argv[1]) == 3:
    scene = "./bookstore/"
    sceneName = scene + vers + "video0/"
    case =  3
    peds = [[184, 229]]

elif int(sys.argv[1]) == 4:
    scene = "./bookstore/"
    sceneName = scene + vers + "video0/"
    case =  4
    peds = [[73,193], [73, 194], [193, 194]]

elif int(sys.argv[1]) == 5:
    scene = "./bookstore/"
    sceneName = scene + vers + "video1/"
    case =  5
    peds = [[237, 261]]

elif int(sys.argv[1]) == 6:
    scene = "./bookstore/"
    sceneName = scene + vers + "video2/"
    case =  6
    peds = [[88, 218]]

elif int(sys.argv[1]) == 7:
    scene = "./coupa/"
    sceneName = scene + vers + "video1/"
    case =  7
    peds = [[71, 73]]

elif int(sys.argv[1]) == 8:
    scene = "./coupa/"
    sceneName = scene + vers + "video2/"
    case =  8
    peds = [[70, 71]]

elif int(sys.argv[1]) == 9:
    scene = "./coupa/"
    sceneName = scene + vers + "video2/"
    case =  9
    peds = [[36, 37]]

elif int(sys.argv[1]) == 10:
    scene = "./coupa/"
    sceneName = scene + vers + "video2/"
    case =  10
    peds = [[42, 65]]

elif int(sys.argv[1]) == 11:
    scene = "./coupa/"
    sceneName = scene + vers + "video2/"
    case =  11
    peds = [[8, 90]]

elif int(sys.argv[1]) == 12:
    scene = "./coupa/"
    sceneName = scene + vers + "video3/"
    case =  12
    peds = [[84, 102]]

elif int(sys.argv[1]) == 13:
    scene = "./coupa/"
    sceneName = scene + vers + "video3/"
    case =  13
    peds = [[51, 86], [51, 85], [85, 86]]

elif int(sys.argv[1]) == 14:
    scene = "./coupa/"
    sceneName = scene + vers + "video0/"
    case =  14
    peds = [[10, 103]]

elif int(sys.argv[1]) == 15:
    scene = "./coupa/"
    sceneName = scene + vers + "video1/"
    case =  15
    peds = [[76, 83]]


elif int(sys.argv[1]) == 16:
    scene = "./coupa/"
    sceneName = scene + vers + "video0/"
    case =  16
    peds = [[86, 115]]

elif int(sys.argv[1]) == 17:
    scene = "./coupa/"
    sceneName = scene + vers + "video1/"
    case =  17
    peds = [[71, 80], [72, 80]]

elif int(sys.argv[1]) == 18:
    scene = "./hyang/"
    sceneName = scene + vers + "video0/"
    case =  18
    peds = [[186, 187]]

elif int(sys.argv[1]) == 19:
    scene = "./bookstore/"
    sceneName = scene + vers + "video5/"
    case =  19
    peds = [[1,87], [76,87], [1,76]]

elif int(sys.argv[1]) == 20:
    scene = "./coupa/"
    sceneName = scene + vers + "video2/"
    case =  20
    peds = [[14, 88]]

elif int(sys.argv[1]) == 21:
    scene = "./quad/"
    sceneName = scene + vers + "video3/"
    case =  21
    peds = [[2,3]]


else:
    print("Incorrect case argument")
# sceneName = "./coupa/video2/"
# case =  15
# peds = [[90, 8]]


csv = pd.read_csv((sceneName + "pos_data_temp.csv"), header = None)

frame_list = csv.iloc[0][:].unique()


# miFile = "MI_tensor_fullres.npy"
miFile = "MI_tensor.npy"

# phiFile_vga_05hdc = "phi_tensor_vga_05hdc.npy"

# phiFile_ga_05hdc = "phi_tensor_ga_05hdc.npy"
# phiFile_va_05hdc = "phi_tensor_va_05hdc.npy"
# phiFile_vg_05hdc = "phi_tensor_vg_05hdc.npy"
# phiFile_vga_05dc = "phi_tensor_vga_05dc.npy"
# phiFile_vga_05hc = "phi_tensor_vga_05hc.npy"
# phiFile_vga_05hd = "phi_tensor_vga_05hd.npy"


# phiFile_a_05hdc = "phi_tensor_a_05hdc.npy"
# phiFile_v_05hdc = "phi_tensor_v_05hdc.npy"
# phiFile_va_05dc = "phi_tensor_va_05dc.npy"
# phiFile_va_05hc = "phi_tensor_va_05hc.npy"
phiFile_va_05hd = "phi_tensor_va_05hd.npy"

# phiFile_a_05hd = "phi_tensor_a_05hd.npy"
# phiFile_v_05hd = "phi_tensor_v_05hd.npy"
# phiFile_va_05d = "phi_tensor_va_05d.npy"
# phiFile_va_05h = "phi_tensor_va_05h.npy"

phiFile_03v_hd = "phi_tensor_03v_hd.npy"
phiFile_03_hd = "phi_tensor_03_hd.npy"
phiFile_03v_d = "phi_tensor_03v_d.npy"
phiFile_03v_h = "phi_tensor_03v_h.npy"

# phiFile_v = "phi_tensor_v.npy"
# phiFile_a = "phi_tensor_a.npy"
# phiFile_d = "phi_tensor_d.npy"
# phiFile_c = "phi_tensor_c.npy"
# phiFile_g = "phi_tensor_g.npy"
# phiFile_h = "phi_tensor_h.npy"


# adiFile_vga_05hdc = "ADI_summed_vga_05hdc.npy"

# adiFile_ga_05hdc = "ADI_summed_ga_05hdc.npy"
# adiFile_va_05hdc = "ADI_summed_va_05hdc.npy"
# adiFile_vg_05hdc = "ADI_summed_vg_05hdc.npy"
# adiFile_vga_05dc = "ADI_summed_vga_05dc.npy"
# adiFile_vga_05hc = "ADI_summed_vga_05hc.npy"
# adiFile_vga_05hd = "ADI_summed_vga_05hd.npy"

# adiFile_va_05dc = "ADI_summed_va_05dc.npy"
# adiFile_v_05dc = "ADI_summed_v_05dc.npy"
# adiFile_vg_05dc = "ADI_summed_vg_05dc.npy"
# adiFile_v_05hdc = "ADI_summed_v_05hdc.npy"

# adiFile_03v_hd = "ADI_summed_03v_hd_Decay_0.npy"
# adiFile_03v_hd = "ADI_summed_03v_hd_Decay_1.0.npy"
# adiFile_03v_hd = "ADI_summed_03v_hd_Decay_2.0.npy"
# adiFile_03v_hd = "ADI_summed_03v_hd_Decay_3.0.npy"
adiFile_03v_hd = "ADI_summed_03v_hd_Decay_5.0.npy"
# adiFile_03v_hd = "ADI_summed_03v_hd_Decay_0.5.npy"




# MIs = np.load((sceneName + "mi/" + miFile), allow_pickle=True)

# phis_vga_05hdc = np.load((sceneName + "phi/" + phiFile_vga_05hdc), allow_pickle=True)

# phis_ga_05hdc = np.load((sceneName + "phi/" + phiFile_ga_05hdc), allow_pickle=True)
# phis_va_05hdc = np.load((sceneName + "phi/" + phiFile_va_05hdc), allow_pickle=True)
# phis_vg_05hdc = np.load((sceneName + "phi/" + phiFile_vg_05hdc), allow_pickle=True)
# phis_vga_05dc = np.load((sceneName + "phi/" + phiFile_vga_05dc), allow_pickle=True)
# phis_vga_05hc = np.load((sceneName + "phi/" + phiFile_vga_05hc), allow_pickle=True)
# phis_vga_05hd = np.load((sceneName + "phi/" + phiFile_vga_05hd), allow_pickle=True)


# phis_a_05hdc = np.load((sceneName + "phi/" + phiFile_a_05hdc), allow_pickle=True)
# phis_v_05hdc = np.load((sceneName + "phi/" + phiFile_v_05hdc), allow_pickle=True)
# phis_va_05dc = np.load((sceneName + "phi/" + phiFile_va_05dc), allow_pickle=True)
# phis_va_05hc = np.load((sceneName + "phi/" + phiFile_va_05hc), allow_pickle=True)
# phis_va_05hd = np.load((sceneName + "phi/" + phiFile_va_05hd), allow_pickle=True)

# phis_a_05hd = np.load((sceneName + "phi/" + phiFile_a_05hd), allow_pickle=True)
# phis_v_05hd = np.load((sceneName + "phi/" + phiFile_v_05hd), allow_pickle=True)
# phis_va_05d = np.load((sceneName + "phi/" + phiFile_va_05d), allow_pickle=True)
# phis_va_05h = np.load((sceneName + "phi/" + phiFile_va_05h), allow_pickle=True)

# phis_03v_hd = np.load((sceneName + "phi/" + phiFile_03v_hd), allow_pickle=True)
# phis_03_hd = np.load((sceneName + "phi/" + phiFile_03_hd), allow_pickle=True)
# phis_03v_d = np.load((sceneName + "phi/" + phiFile_03v_d), allow_pickle=True)
# phis_03v_h = np.load((sceneName + "phi/" + phiFile_03v_h), allow_pickle=True)


# phis_v = np.load((sceneName + "phi/" + phiFile_v), allow_pickle=True)
# phis_a = np.load((sceneName + "phi/" + phiFile_a), allow_pickle=True)
# phis_d = np.load((sceneName + "phi/" + phiFile_d), allow_pickle=True)
# phis_c = np.load((sceneName + "phi/" + phiFile_c), allow_pickle=True)
# phis_g = np.load((sceneName + "phi/" + phiFile_g), allow_pickle=True)
# phis_h = np.load((sceneName + "phi/" + phiFile_h), allow_pickle=True)

# ADIs_vga_05hdc = np.load((sceneName + "adi/" + adiFile_vga_05hdc), allow_pickle=True)

# ADIs_ga_05hdc = np.load((sceneName + "adi/" + adiFile_ga_05hdc), allow_pickle=True)
# ADIs_va_05hdc = np.load((sceneName + "adi/" + adiFile_va_05hdc), allow_pickle=True)
# ADIs_vg_05hdc = np.load((sceneName + "adi/" + adiFile_vg_05hdc), allow_pickle=True)
# ADIs_vga_05dc = np.load((sceneName + "adi/" + adiFile_vga_05dc), allow_pickle=True)
# ADIs_vga_05hc = np.load((sceneName + "adi/" + adiFile_vga_05hc), allow_pickle=True)
# ADIs_vga_05hd = np.load((sceneName + "adi/" + adiFile_vga_05hd), allow_pickle=True)

# ADIs_va_05dc = np.load((sceneName + "adi/" + adiFile_va_05dc), allow_pickle=True)
# ADIs_v_05dc = np.load((sceneName + "adi/" + adiFile_v_05dc), allow_pickle=True)
# ADIs_vg_05dc = np.load((sceneName + "adi/" + adiFile_vg_05dc), allow_pickle=True)
# ADIs_vga_05hdc = np.load((sceneName + "adi/" + adiFile_vga_05hdc), allow_pickle=True)
# ADIs_v_05hdc = np.load((sceneName + "adi/" + adiFile_v_05hdc), allow_pickle=True)

ADIs_03v_hd = np.load((sceneName + "adi/" + adiFile_03v_hd), allow_pickle=True)


# visualize(peds, csv, frame_list, "MI", MIs, sceneName, miFile[:-4], case, vers, scene)

# visualize(peds, csv, frame_list, "phi", phis_vga_05hdc, sceneName, phiFile_vga_05hdc[:-4], case, vers, scene)

# visualize(peds, csv, frame_list, "phi", phis_ga_05hdc, sceneName, phiFile_ga_05hdc[:-4], case, vers, scene)
# visualize(peds, csv, frame_list, "phi", phis_va_05hdc, sceneName, phiFile_va_05hdc[:-4], case, vers, scene)
# visualize(peds, csv, frame_list, "phi", phis_vg_05hdc, sceneName, phiFile_vg_05hdc[:-4], case, vers, scene)
# visualize(peds, csv, frame_list, "phi", phis_vga_05dc, sceneName, phiFile_vga_05dc[:-4], case, vers, scene)
# visualize(peds, csv, frame_list, "phi", phis_vga_05hc, sceneName, phiFile_vga_05hc[:-4], case, vers, scene)
# visualize(peds, csv, frame_list, "phi", phis_vga_05hd, sceneName, phiFile_vga_05hd[:-4], case, vers, scene)


# visualize(peds, csv, frame_list, "phi", phis_a_05hdc, sceneName, phiFile_a_05hdc[:-4], case, vers, scene)
# visualize(peds, csv, frame_list, "phi", phis_v_05hdc, sceneName, phiFile_v_05hdc[:-4], case, vers, scene)
# visualize(peds, csv, frame_list, "phi", phis_va_05dc, sceneName, phiFile_va_05dc[:-4], case, vers, scene)
# visualize(peds, csv, frame_list, "phi", phis_va_05hc, sceneName, phiFile_va_05hc[:-4], case, vers, scene)
# visualize(peds, csv, frame_list, "phi", phis_va_05hd, sceneName, phiFile_va_05hd[:-4], case, vers, scene)

# visualize(peds, csv, frame_list, "phi", phis_a_05hd, sceneName, phiFile_a_05hd[:-4], case, vers, scene)
# visualize(peds, csv, frame_list, "phi", phis_v_05hd, sceneName, phiFile_v_05hd[:-4], case, vers, scene)
# visualize(peds, csv, frame_list, "phi", phis_va_05d, sceneName, phiFile_va_05d[:-4], case, vers, scene)
# visualize(peds, csv, frame_list, "phi", phis_va_05h, sceneName, phiFile_va_05h[:-4], case, vers, scene)


# visualize(peds, csv, frame_list, "phi", phis_03v_hd, sceneName, phiFile_03v_hd[:-4], case, vers, scene)
# visualize(peds, csv, frame_list, "phi", phis_03_hd, sceneName, phiFile_03_hd[:-4], case, vers, scene)
# visualize(peds, csv, frame_list, "phi", phis_03v_d, sceneName, phiFile_03v_d[:-4], case, vers, scene)
# visualize(peds, csv, frame_list, "phi", phis_03v_h, sceneName, phiFile_03v_h[:-4], case, vers, scene)


# visualize(peds, csv, frame_list, "phi", phis_v, sceneName, phiFile_v[:-4], case, vers, scene)
# visualize(peds, csv, frame_list, "phi", phis_a, sceneName, phiFile_a[:-4], case, vers, scene)
# visualize(peds, csv, frame_list, "phi", phis_d, sceneName, phiFile_d[:-4], case, vers, scene)
# visualize(peds, csv, frame_list, "phi", phis_c, sceneName, phiFile_c[:-4], case, vers, scene)
# visualize(peds, csv, frame_list, "phi", phis_g, sceneName, phiFile_g[:-4], case, vers, scene)
# visualize(peds, csv, frame_list, "phi", phis_h, sceneName, phiFile_h[:-4], case, vers, scene)

# # visualize(peds, csv, frame_list, "ADI", ADIs_vga_05hdc, sceneName, adiFile_vga_05hdc[:-4], case, vers, scene)
# # visualize(peds, csv, frame_list, "ADI", ADIs_ga_05hdc, sceneName, adiFile_ga_05hdc[:-4], case, vers, scene)
# # visualize(peds, csv, frame_list, "ADI", ADIs_va_05hdc, sceneName, adiFile_va_05hdc[:-4], case, vers, scene)
# # visualize(peds, csv, frame_list, "ADI", ADIs_vg_05hdc, sceneName, adiFile_vg_05hdc[:-4], case, vers, scene)
# # visualize(peds, csv, frame_list, "ADI", ADIs_vga_05dc, sceneName, adiFile_vga_05dc[:-4], case, vers, scene)
# # visualize(peds, csv, frame_list, "ADI", ADIs_vga_05hc, sceneName, adiFile_vga_05hc[:-4], case, vers, scene)
# # visualize(peds, csv, frame_list, "ADI", ADIs_vga_05hd, sceneName, adiFile_vga_05hd[:-4], case, vers, scene)

# # visualize(peds, csv, frame_list, "ADI", ADIs_va_05dc, sceneName, adiFile_va_05dc[:-4], case, vers, scene)
# # visualize(peds, csv, frame_list, "ADI", ADIs_v_05dc, sceneName, adiFile_v_05dc[:-4], case, vers, scene)
# # visualize(peds, csv, frame_list, "ADI", ADIs_vg_05dc, sceneName, adiFile_vg_05dc[:-4], case, vers, scene)
# # visualize(peds, csv, frame_list, "ADI", ADIs_v_05hdc, sceneName, adiFile_v_05hdc[:-4], case, vers, scene)

visualize(peds, csv, frame_list, "ADI", ADIs_03v_hd, sceneName, adiFile_03v_hd[:-4], case, vers, scene)
























# peds = [[69, 10],  [69, 16], [69,68], [69, 102], [69, 103], [69, 104]]
# ped = [[102,103], [102,68], [102, 104], [102,10]]
# peds = [[69, 104]]
# peds = [[53, 55], [54,55], [53,54]]
# peds = [[216, 237]]
# peds = [[88, 167]]
# peds = [[88, 167]]
# peds = [[5, 25]]
# peds = [[36, 37]]
# peds = [[42, 65]]
# peds = [[90, 8], [90,80], [90,87]]
# peds = [[51, 110]]
# peds = [[51,86], [51, 85], [85, 86]]
# peds = [[51,86]]
# peds = [[86, 115]] ### Same with fullres
# peds = [[45, 74]]
# peds = [[48, 74]] ### video1 with fullres
# peds = [[70,71]]
# peds = [[16,54]] ### video2 with fullres
# peds = [[84, 102]] ### Same with fullres
# peds = [[71,73]]