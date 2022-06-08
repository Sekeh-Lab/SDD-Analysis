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





def visualize(peds, csv, frame_list, measure, measureVals, sceneName, title, case, vers, scene):
    # csv = csv.transpose()
    maxVal = 0
    cmap = cm.get_cmap('jet')
    
    
    for i in range(0, len(measureVals[0])):
        for j in range(0, len(measureVals[0])):
            if measureVals[i][j] != 0:
                for frame in measureVals[i][j]:
                    if frame[1] > maxVal:
                        maxVal = frame[1]

        

    imgplt = plt.imread((sceneName + "background.png"))
    range_x = imgplt.shape[1]
    range_y = imgplt.shape[0]


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
            ### Store the pair of trajectory data for agents i and j
            traj_data = np.asarray([traj_shared_i.to_numpy(), traj_shared_j.to_numpy()])
            

            plotvals_i = pd.DataFrame.from_records(vals_i, columns = ["frame", "measure"])
            plotvals_j = pd.DataFrame.from_records(vals_j, columns = ["frame", "measure"])
            ### At this point traj_data contains the shared trajectory to be visualized
    
            plotvals_i = plotvals_i[(plotvals_i.iloc[:,0].astype(int)).between(int(traj_shared_i.iloc[0,0]), int(traj_shared_i.iloc[-1,0]))]
            plotvals_j = plotvals_j[(plotvals_j.iloc[:,0].astype(int)).between(int(traj_shared_j.iloc[0,0]), int(traj_shared_j.iloc[-1,0]))]

            label_i = "o"
            label_j = "^"

            ### Something like this to get all frames in the shared trajectory
            for frame, frame_outer in enumerate(traj_data[0][-1:,0]):

                file = sceneName + "background.png"

                img = plt.imread(file, format = 'png')
                plt.imshow(img)

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

                legend_elements = [Line2D([0], [0], marker='o', linestyle = "None", color='black', label='Agent {}'.format(str(i)), markerfacecolor='black', markersize=10),
                           Line2D([0], [0], marker='^', linestyle = "None", color='black', label='Agent {}'.format(str(j)), markerfacecolor='black', markersize=10)]

                plt.legend(handles=legend_elements, prop = {"size":10, "weight":"bold"}, loc = (.125,0.55))
                plt.savefig("./interactions/" + sceneName[2:] + "case_{}/{}/".format(case, measure) + "test_i{}_j{}_{}/".format(str(i), str(j), str(title)) + "test_i{}_j{}_frame{}.jpg".format(str(i), str(j), str(frame_outer)), dpi = 200, bbox_inches = "tight")
                plt.clf()


                plt.colorbar(mpl.cm.ScalarMappable(norm = mpl.colors.Normalize(vmin=0, vmax = maxVal), cmap = "gnuplot2"), ticks = np.linspace(0, maxVal, 10, endpoint=True))
                plt.savefig("./interactions/" + sceneName[2:] + "case_{}/{}/".format(case, measure) + "test_i{}_j{}_{}/".format(str(i), str(j), str(title)) + "bar_i{}_j{}_frame{}.jpg".format(str(i), str(j), str(frame_outer)), dpi = 200, bbox_inches = "tight")
                plt.clf()

                plt.figure(figsize = (5,5), linewidth = 10)
                ax = plt.gca()
                ax.set_aspect(15)
                plt.rc('ytick',labelsize = 12)
                plt.rc('xtick',labelsize = 12)
                plt.locator_params(axis="x", nbins=6)
                plt.locator_params(axis="y", nbins=5)
                plt.plot(plotvals_i.iloc[:,0], plotvals_i.iloc[:,1], c = "red", linewidth = 3, label = "Agent {}".format(str(i)))
                plt.plot(plotvals_i.iloc[:,0], plotvals_j.iloc[:,1], c = "blue", linewidth = 3, linestyle = "dashed", label = "Agent {}".format(str(j)))
                plt.xlabel("Frame (25 fps)", fontsize = 13, fontweight = "bold")
                plt.ylabel("AIM Value", fontsize = 13, fontweight = "bold")
                plt.legend(prop = {"size":12, "weight":"bold"})
                plt.savefig("./interactions/" + sceneName[2:] + "case_{}/{}/".format(case, measure) + "test_i{}_j{}_{}/".format(str(i), str(j), str(title)) + "graph_i{}_j{}_frame{}.jpg".format(str(i), str(j), str(frame_outer)), dpi = 200, bbox_inches = "tight")
                plt.clf()


        

print("started")

peds = []
case = 100
sceneName = ""
scene = ""

vers = ""
if sys.argv[2] == "pixel":
    vers = "pixel/"
elif sys.argv[2] == "corrected":
    vers = "corrected/"
else:
    print("incorrect version input")




if int(sys.argv[1]) == 0:
    scene = "Scene1/"
    sceneName = "./" + vers + scene + "0/"
    case = 0
    peds = [[1, 7], [1,6]]
elif int(sys.argv[1]) == 1:
    scene = "Scene1/"
    sceneName = "./" + vers + scene + "0/"
    case = 1
    peds = [[6,9],[6,10], [9,10]]
elif int(sys.argv[1]) == 2:
    scene = "Scene1/"
    sceneName = "./" + vers + scene + "0/"
    case = 2
    peds = [[30,32],[30,33]]
elif int(sys.argv[1]) == 3:
    scene = "Scene1/"
    sceneName = "./" + vers + scene + "0/"
    case = 3
    peds = [[30,34],[30,35],[34,35]]
elif int(sys.argv[1]) == 4:
    scene = "Scene1/"
    sceneName = "./" + vers + scene + "0/"
    case = 4
    peds = [[34,41],[35,41],[41,45]]
elif int(sys.argv[1]) == 5:
    scene = "Scene1/"
    sceneName = "./" + vers + scene + "0/"
    case = 5
    peds = [[46,48]]
elif int(sys.argv[1]) == 6:
    scene = "Scene1/"
    sceneName = "./" + vers + scene + "0/"
    case = 6
    peds = [[50,51]]
elif int(sys.argv[1]) == 7:
    scene = "Scene1/"
    sceneName = "./" + vers + scene + "0/"
    case = 7
    peds = [[218,221]]
elif int(sys.argv[1]) == 8:
    scene = "Scene1/"
    sceneName = "./" + vers + scene + "0/"
    case = 8
    peds = [[34,41], [34,45], [41,45]]
elif int(sys.argv[1]) == 9:
    scene = "Scene1/"
    sceneName = "./" + vers + scene + "1/"
    case = 9
    peds = [[238,239]]
elif int(sys.argv[1]) == 10:
    scene = "Scene1/"
    sceneName = "./" + vers + scene + "1/"
    case = 10
    peds = [[128,133], [128,137]]



else:
    print("Incorrect case argument")


csv = pd.read_csv((sceneName + "pos_data_temp.csv"), header = None)

frame_list = csv.iloc[0][:].unique()



### Hard coded to allow selecting of which metric and formula to visualize

# miFile = "MI_tensor_fullres.npy"
# miFile = "MI_tensor.npy"

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
# phiFile_va_05hd = "phi_tensor_va_05hd.npy"

# phiFile_a_05hd = "phi_tensor_a_05hd.npy"
# phiFile_v_05hd = "phi_tensor_v_05hd.npy"
# phiFile_va_05d = "phi_tensor_va_05d.npy"
# phiFile_va_05h = "phi_tensor_va_05h.npy"

# phiFile_03v_hd = "phi_tensor_03v_hd.npy"
# phiFile_03_hd = "phi_tensor_03_hd.npy"
# phiFile_03v_d = "phi_tensor_03v_d.npy"
# phiFile_03v_h = "phi_tensor_03v_h.npy"

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

# adiFile_03v_hd = "ADI_summed_03v_hd.npy"

adiFile_03v_hd = "ADI_summed_03v_hd_Decay_2.0.npy"



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
