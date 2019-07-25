# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 20:12:36 2019

@author: Johnny Tsao
"""
import numpy as np
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
cwd = os.getcwd()
def read_input(filename):
    data = open(cwd+"\\" + filename,"r")
    data_split = data.read().split('\n')
    iter_list = []
    maxDif_list = []
    it_list = []
    for i in range(1,len(data_split),1):
        row = data_split[i]
        elem = row.split(',')
        print(elem)
        if(len(elem) >= 3):
            iter_list.append(elem[0])
            maxDif_list.append(elem[1])
            it_list.append(elem[2])
    iter_array = np.array(iter_list,dtype = int)
    maxDif_array = np.array(maxDif_list,dtype = float)
    it_array = np.array(it_list,dtype = int)
    l1=np.log(iter_array)
    l2=np.log(maxDif_array)
    plt.title("Convergence plot for Dirichlet boundary (test case 2)")
    plt.plot(l1,l2)
#    plt.plot(np.linspace(min(l1),max(l1),20),np.linspace(max(l2),max(l2)-2*(max(l1)-min(l1)),20))
    
    x=l1.reshape((-1,1))
    model = LinearRegression().fit(x, l2)
    plt.plot(x,model.predict(x),'--')
    plt.plot(l1,l2,"r.")
    plt.show()
    plt.xlabel("log(N_grid)")
    plt.ylabel("log(error)")
    plt.legend(["result", "slope = %4g" % model.coef_[0]])
read_input("Convergence_result.csv")


