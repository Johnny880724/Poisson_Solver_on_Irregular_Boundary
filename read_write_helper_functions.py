# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 20:52:34 2019

@author: Johnny Tsao
"""

import os
cwd = os.getcwd()
def write_float_data(filename,arrays):
    file = open(cwd + "\\data\\" + filename + ".csv","w+")
    for array in arrays:
        for elem in array:
            file.write(str(elem))
            file.write(",")
        file.write("\n")
    file.close()
test = [[1,2,3],[2,3,4,5],[1.2,3,2]]
write_float_data("test", test)

def read_float_data(filename):
    file = open(cwd + "\\data\\" + filename + ".csv","r")
    data = file.read().split('\n')
    data_output = []
    for array in data:
        elem = array.split(',')
        elem.remove('')
        if(len(elem) > 0):
            data_output.append(elem)
    return data_output

        
data = read_float_data("test")
