import numpy as np
import pandas as pd
import os
from typing import List
import matplotlib.pyplot as plt
from scipy import signal

# enumerating temps
temperatures = [100, 105, 10, 110, 115, 120, 125, 130, 135, 13, 140, 145, 150, 155, 15, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 20, 210, 215, 220, 225, 230, 235, 240, 250, 25, 260, 270, 27, 280, 290, 300, 30, 35, 40, 45, 50, 53, 55, 57, 5, 60, 63, 65, 70, 75, 80, 85, 8, 90, 95, 245]
temperatures.sort()
n_temps = len(temperatures)

def Temp(t: int) -> int:
    """
    This function returns the index of the temp
    """
    return temperatures.index(t)

def load_data(folders: List[str]) -> np.ndarray:
    # Dimentions/axes: power, polarization, temp, column (reflectivity, kerr, delay), points
    data = np.empty((3, 3, 61, 3, 3200))

    # temps = []  # for finding uniqe temps

    # file_counter = 0  # for counting total number of files

    for folder in folders:
        for root, subdirs, files in os.walk(folder):
            for fname in files:
                if fname[:9] == "PumpProbe":
                    # file_counter += 1  # for counting total number of files

                    splitted_str = fname.split("_")

                    ########### power ############
                    power = int(splitted_str[5][:2])
                    if power == 40:
                        power = 25

                    if power == 10:
                        axis_power = 0
                    elif power == 25:
                        axis_power = 1
                    elif power == 48:
                        axis_power = 2
                    ##############################

                    ############ temp ############
                    temp = int(splitted_str[4][:-1])
                    # if temp not in temps:  # for finding uniqe temps
                    # temps.append(temp)

                    axis_temp = temperatures.index(temp)
                    ##############################

                    ############ polarization ############
                    polarization = splitted_str[-1]

                    if polarization == "right":
                        axis_pol = 0
                    elif polarization == "left":
                        axis_pol = 1
                    elif polarization == "lin":
                        axis_pol = 2
                    ######################################

                    measurement = pd.read_csv(os.path.join(root, fname), header=9)

                    data[axis_power, axis_pol, axis_temp, 0, :] = measurement["Delay [ps]"].to_numpy()
                    data[axis_power, axis_pol, axis_temp, 1, :] = measurement["Rel_Delta_X2"].to_numpy()
                    data[axis_power, axis_pol, axis_temp, 2, :] = measurement["Rel_Delta_X"].to_numpy()

                    print(axis_power, axis_pol, axis_temp)

                    return data


def plot_single(full_data, y_axis, power, pol, temp, x_axis=0):
    """
    This function takes the relevent axes and plots a single line
    """

    if y_axis == "ref":
        y_axis = 1
    elif y_axis == "kerr":
        y_axis = 2

    x = full_data[power, pol, temp, x_axis, :]
    y = full_data[power, pol, temp, y_axis, :]

    plt.plot(x, y)

a = pd.read_csv("Data/polarization temp run10mW/PumpProbe_Kerr_sample3_spot3_110K_10mW_long_left",header = 9).to_numpy()
b = np.flip(a, axis=0)
print(a)
print(a["Delay [ps]"])
print(a.to_numpy()[:,0])

#data = load_data([r"Data\polarization temp run10mW", r"Data\polarization temp run25mW", r"Data\polarization temp run48mW"])

#plot_single(data, y_axis="ref", power=0, pol=0, temp=temperatures.index(5))

