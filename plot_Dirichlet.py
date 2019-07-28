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
from sklearn.linear_model import LinearRegression

#N_grid_array = [32,64,128,256]
#fig = plt.figure()
#ax = fig.gca()
#ax.set_title("error - iterations plot")
#ax.set_xlabel("iterations")
#ax.set_ylabel("log(error)")
#for N_grid in N_grid_array:
#    data = rw.read_float_data("\\Dirichlet\\Dirichlet_"+str(N_grid))
#    it_array = np.array(data[0],dtype=int)+1
#    maxDif_array =np.array(data[1],dtype = float)
#    ax.plot(it_array,np.log(maxDif_array),marker='.' )
#legend_array = [("N = " + str(i) )for i in N_grid_array]
#ax.legend(legend_array)
def plot_it_error(plot_type):
    #plot Dirichlet, Neumann, Robin
    #it-error plot
    fig = plt.figure()
    ax = fig.gca()
    ax.set_title(plot_type + " error - iterations plot")
    ax.set_xlabel("iterations")
    ax.set_ylabel("log(error)")
    convergence_data = rw.read_float_data("\\source_term_method\\"+plot_type+"\\"+plot_type+"_convergence")
    N_grid_array = np.array(convergence_data[0],dtype = int)
    for N_grid in N_grid_array:
        data = rw.read_float_data("\\source_term_method\\"+plot_type+"\\"+plot_type+"_"+str(N_grid))
        iter_array = np.array(data[0],dtype=int)+1
        maxDif_array = np.array(data[1],dtype = float)
        L2Dif_array = np.array(data[2],dtype = float)
        ax.plot(iter_array,np.log(maxDif_array),marker='.' )
    legend_array = [("N = " + str(i) )for i in N_grid_array]
    ax.legend(legend_array)
    
def plot_convergence(plot_type):
    fig = plt.figure()
    ax = fig.gca()
    ax.set_title(plot_type + " convergence plot")
    ax.set_xlabel("log(grid_size)")
    ax.set_ylabel("log(error)")
    convergence_data = rw.read_float_data("\\source_term_method\\"+plot_type+"\\"+plot_type+"_convergence")
    N_grid_array = np.array(convergence_data[0],dtype = int)
    error_array = np.array(convergence_data[1],dtype = float)
    
    l1=np.log(N_grid_array)
    l2=np.log(error_array)
    plt.plot(l1,l2,marker='.')
    plt.plot(np.linspace(min(l1),max(l1),20),np.linspace(max(l2),max(l2)-2*(max(l1)-min(l1)),20))
    plt.xlabel("log(N_grid)")
    plt.ylabel("log(error)")
    plt.title("convergence plot for "+plot_type +" boundary")
    
    x=l1.reshape((-1,1))
    model = LinearRegression().fit(x, l2)
    plt.plot(x,model.predict(x),'--')
    plt.legend(["result","theoretical line: slope = -2" ,"fitted line: slope = %g" % model.coef_[0] ])
    #the slope is -1.18543593
if(__name__ == "__main__"):
    plt.close("all")
    plot_it_error("Dirichlet")
    plot_it_error("Neumann")
    plot_it_error("Robin")
    plot_convergence("Dirichlet")
    plot_convergence("Neumann")
    plot_convergence("Robin")