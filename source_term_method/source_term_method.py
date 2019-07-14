# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 14:25:56 2019

@author: Johnny Tsao
"""
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import poisson_helper_functions as hf

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy import linalg as la

#adding to denominator to avoid 0/0 error
singular_null = 1.0e-17

def setup_grid(N_grid_val = 101):
    # grid dimension
    grid_min = -1.
    grid_max = 1.
    
    global N_grid
    N_grid = N_grid_val
    # grid spacing
    global h
    h = (grid_max - grid_min) / (N_grid - 1) 
    
    # define arrays to hold the x and y coordinates
    xy = np.linspace(grid_min,grid_max,N_grid)
    global xmesh, ymesh
    xmesh, ymesh = np.meshgrid(xy,xy)
    
    # solution
    global u_init
    u_init = np.zeros_like(xmesh)

#discretization of Step, Delta functions
def I(phi):
    return np.heaviside(phi,1)*phi

def J(phi):
    return 1./2 * np.heaviside(phi,1)*phi**2

def K(phi):
    return 1./6 * np.heaviside(phi,1)*phi**3

#Characteristic function of N1
##Chi is 1 if in N1
##       0 if not in N1
def Chi(phi):
    ret = np.zeros_like(phi)
    ret[:-1,:] += np.heaviside(-phi[:-1,:]*phi[ 1:,:],1)
    ret[ 1:,:] += np.heaviside(-phi[:-1,:]*phi[ 1:,:],1)
    ret[:,:-1] += np.heaviside(-phi[:,:-1]*phi[ :,1:],1)
    ret[:, 1:] += np.heaviside(-phi[:,:-1]*phi[ :,1:],1)
    return np.heaviside(ret,0)

def H(phi,h):
#    I_mat = I(phi)
    J_mat = J(phi)
    K_mat = K(phi)
    first_term = hf.laplace(J_mat,h,h) / (hf.abs_grad(phi,h,h)**2 + singular_null)
    first_term -= (hf.laplace(K_mat,h,h) - J_mat*hf.laplace(phi,h,h))*hf.laplace(phi,h,h) / (hf.abs_grad(phi,h,h)**4 + singular_null)
    second_term = np.heaviside(phi,1)
    return Chi(phi) * first_term + (1-Chi(phi)) * second_term

def delta(phi,h):
    I_mat = I(phi)
    J_mat = J(phi)
#    K_mat = K(phi)
    first_term = hf.laplace(I_mat,h,h) / (hf.abs_grad(phi,h,h)**2 + singular_null)
    first_term -= (hf.laplace(J_mat,h,h) - I_mat*hf.laplace(phi,h,h))*hf.laplace(phi,h,h) / (hf.abs_grad(phi,h,h)**4 + singular_null)
    return Chi(phi) * first_term

def rhs(x,y):
    return -np.ones_like(x)

def get_source(a, b, mesh, lvl_func_):
    xmesh, ymesh = mesh
    h = xmesh[0,1]-xmesh[0,0]
    phi = lvl_func_(xmesh,ymesh)
    a_eff = a + (hf.norm_grad(a,(xmesh,ymesh),lvl_func_) - b) * phi / (hf.abs_grad(phi,h,h)+singular_null)
    S_mat = np.zeros_like(xmesh)
    H_mat = H(phi,h)
    f_mat = rhs(xmesh, ymesh)
    plt.matshow(f_mat)
    plt.colorbar()
    S_mat = hf.laplace(a_eff * H_mat,h,h) - H_mat * hf.laplace(a_eff,h,h) + H_mat * f_mat
    return S_mat

plt.close("all")
setup_grid(101)
r0=0.8
phi = -np.array(hf.XYtoR(xmesh,ymesh) - r0)
#plt.matshow(Chi(phi))
#plt.colorbar()
#plt.matshow(delta(phi,h))
#plt.colorbar()
plt.matshow(H(phi,h))
plt.colorbar()


x0 = 0.0
y0 = 0.0
def level(x,y):
    return -np.array(hf.XYtoR(x,y) - r0)

global lvl_func
lvl_func = level
def desired_func(x, y):
    return (1 - 0.25 * ((x - x0)**2 + (y - y0)**2))
u_desired = desired_func(xmesh,ymesh)
u_n_desired = hf.norm_grad(u_desired,(xmesh,ymesh),lvl_func)
a_mesh =  u_desired
b_mesh =  u_n_desired

    

#level grid
phi = lvl_func(xmesh,ymesh)
h = xmesh[0,1]-xmesh[0,0]
#    u = u_init_
u=np.zeros_like(xmesh)

# normal
r_temp = hf.XYtoR(xmesh, ymesh) + 10**-17
n1, n2 = xmesh/r_temp, ymesh/r_temp

# the source on the rhs
#source = np.zeros_like(u)
source = np.zeros_like(xmesh)


def get_theta(u_cur,u_next):
    return np.abs(u_cur)/(np.abs(u_cur)+np.abs(u_next))

source = get_source(a_mesh, b_mesh, (xmesh, ymesh),lvl_func)
plt.matshow(source)
plt.colorbar()
#record data
u_prev = u
global iterNum_record
iterNum_record = 0
maxIterNum_ = 200000
for i in range(maxIterNum_):
    iterNum_record += 1
    
    
    # enforce boundary condition
    #  sol = solution1(xmesh, ymesh)
    u[ 0, :] = np.zeros_like(u[ 0, :])
    u[-1, :] = np.zeros_like(u[-1, :])
    u[ :, 0] = np.zeros_like(u[ :, 0])
    u[ :,-1] = np.zeros_like(u[ :,-1])

    u_new = np.copy(u)

    # update u according to Jacobi method formula
    # https://en.wikipedia.org/wiki/Jacobi_method
    
    del_u = u[1:-1,2:] + u[1:-1,0:-2] + u[2:,1:-1] + u[0:-2,1:-1]
    u_new[1:-1,1:-1] = -h**2/4 * (source[1:-1,1:-1] - del_u/h**2)
    u = u_new
    if(i % int(maxIterNum_/100) < 0.1):
        u_cur = u
        maxDif = np.max(np.abs(u_cur - u_prev)) / np.max(np.abs(u_cur))
        L2Dif = hf.L_n_norm(np.abs(u_cur - u_prev)) / hf.L_n_norm(u_cur)
        # check convergence and print process
        check_convergence_rate = 10**-11
        if(i % int(maxIterNum_/100) < 0.1):
            u_cur = u
            maxDif = np.max(np.abs(u_cur - u_prev)) / np.max(np.abs(u_cur))
            L2Dif = hf.L_n_norm(np.abs(u_cur - u_prev)) / hf.L_n_norm(u_cur)
            if(L2Dif < check_convergence_rate):
                break;
            else:
                u_prev = u_cur
            sys.stdout.write("\rProgress: %4g%%" % (i *100.0/maxIterNum_))
            sys.stdout.flush()
        sys.stdout.write("\rProgress: %4g%%" % (i *100.0/maxIterNum_))
        sys.stdout.flush()
    
print("")
plt.matshow(u)
plt.colorbar()