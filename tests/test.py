# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:03:27 2019

@author: Johnny Tsao
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy import linalg as la

grid_min = -1.
grid_max = 1.
N_grid = 101
h = (grid_max - grid_min) / (N_grid- 1) 

# define arrays to hold the x and y coordinates
xy = np.linspace(grid_min,grid_max,N_grid)
xmesh, ymesh = np.meshgrid(xy,xy)

# solution
u = np.zeros_like(xmesh)

u = xmesh**2 + ymesh**2
print(xmesh[0,:])

plt.close("all")
plt.matshow(xmesh)
plt.colorbar()