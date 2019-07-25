# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 00:16:57 2019

@author: Johnny Tsao
"""


import poisson_helper_functions as hf

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy import linalg as la
import matplotlib.animation as animation
import read_write_helper_functions as rw

N_grid_array = [32,64,128,256]
fig = plt.figure()
ax = fig.gca()
ax.set_title("error - iterations plot")
ax.set_xlabel("iterations")
ax.set_ylabel("log(error)")
for N_grid in N_grid_array:
    data = rw.read_float_data("\\Dirichlet\\Dirichlet_"+str(N_grid))
    it_array = np.array(data[0],dtype=int)+1
    maxDif_array =np.array(data[1],dtype = float)
    ax.plot(it_array,np.log(maxDif_array),marker='.' )
legend_array = [("N = " + str(i) )for i in N_grid_array]
ax.legend(legend_array)