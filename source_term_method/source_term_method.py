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

def source(a, b, mesh, phi):
    xmesh, ymesh = mesh
    h = xmesh[0,1]-xmesh[0,0]
    S_mat = np.zeros_like(xmesh)
    H_mat = H(phi,h)
    S += hf.laplace(b * H_mat)

def poisson_jacobi_solver(u_init_, maxIterNum_, mesh_, beta_, rhs_func_, lvl_func_, jmp_func_):
    xmesh, ymesh = mesh_
    beta_p, beta_m = beta_
    #level grid
    phi = lvl_func_(xmesh,ymesh)
    h = xmesh[0,1]-xmesh[0,0]
#    u = u_init_
    u=np.zeros_like(xmesh)
    
    # normal
    r_temp = hf.XYtoR(xmesh, ymesh) + 10**-17
    n1, n2 = xmesh/r_temp, ymesh/r_temp
    
    # the source on the rhs
    #source = np.zeros_like(u)
    source = np.zeros_like(xmesh)
    
    isOut = np.array(np.greater(phi,0.0),dtype = int)
    source += (1.0 - isOut) * rhs_func_(xmesh, ymesh)
    a_mesh, b_mesh = jmp_func_(xmesh, ymesh, u_init_)
    
    def get_theta(u_cur,u_next):
        return np.abs(u_cur)/(np.abs(u_cur)+np.abs(u_next))
    
    def get_effective_beta(beta_cur,beta_next,theta):
        ret = beta_cur * beta_next / (beta_next * theta + beta_cur * (1-theta))
        return ret
    
    
    beta_x = np.zeros_like(xmesh[:,1:])
    beta_y = np.zeros_like(ymesh[1:,:])
    
    
    isOutx1 = np.array(np.greater(phi[:,1:],0.0),dtype = int)
    isOutx2 = np.array(np.greater(phi[:,:-1],0.0),dtype = int)
    isOuty1 = np.array(np.greater(phi[1:,:],0.0),dtype = int)
    isOuty2 = np.array(np.greater(phi[:-1,:],0.0),dtype = int)
    
#    step is 1 if k+1 is out and k is in
#    step is -1 if k is out and k+1 is in
#    step is 0 if both out or in
    xstep = isOutx1 - isOutx2
    ystep = isOuty1 - isOuty2
    
    xstep_p = np.array(np.greater( xstep,0.0),dtype = int)
    xstep_m = np.array(np.greater(-xstep,0.0),dtype = int)
    
    ystep_p = np.array(np.greater( ystep,0.0),dtype = int)
    ystep_m = np.array(np.greater(-ystep,0.0),dtype = int)
    
    theta_x = get_theta(phi[:,:-1],phi[:,1:])
    theta_y = get_theta(phi[:-1,:],phi[1:,:])
    
    beta_eff_x = np.zeros_like(beta_x)
    beta_eff_y = np.zeros_like(beta_y)
    
    beta_eff_x += xstep_p * get_effective_beta(beta_m,beta_p,theta_x)
    beta_eff_x += xstep_m * get_effective_beta(beta_p,beta_m,theta_x)
    beta_eff_y += ystep_p * get_effective_beta(beta_m,beta_p,theta_y)
    beta_eff_y += ystep_m * get_effective_beta(beta_p,beta_m,theta_y)
    
    a_jump_x = a_mesh[:,:-1] * (1-theta_x) + a_mesh[:,1:] * theta_x
    a_jump_y = a_mesh[:-1,:] * (1-theta_y) + a_mesh[1:,:] * theta_y
    
    b_jump_x = b_mesh[:,:-1]*(1-theta_x)*n1[:,:-1] + b_mesh[:,1:]*theta_x*n1[:,1:]
    b_jump_y = b_mesh[:-1,:]*(1-theta_y)*n2[:-1,:] + b_mesh[1:,:]*theta_y*n2[1:,:]
#    first additional term
    source[:,:-1] += a_jump_x * beta_eff_x*xstep/h**2
    source[:, 1:] += -a_jump_x * beta_eff_x*xstep/h**2
    source[:-1,:] += a_jump_y * beta_eff_y*ystep/h**2
    source[ 1:,:] += -a_jump_y * beta_eff_y*ystep/h**2
    
    beta_x_k   = ((beta_p + beta_m) + (beta_p - beta_m)*xstep)/2
    beta_x_kp1 = ((beta_p + beta_m) - (beta_p - beta_m)*xstep)/2
    beta_y_k   = ((beta_p + beta_m) + (beta_p - beta_m)*ystep)/2
    beta_y_kp1 = ((beta_p + beta_m) - (beta_p - beta_m)*ystep)/2
    
#    second additional term
    source[:,:-1] += (b_jump_x * beta_eff_x * xstep / h) * ((1-theta_x)/beta_x_k)
    source[:, 1:] +=  (b_jump_x * beta_eff_x * xstep / h) * (theta_x/beta_x_kp1)
    source[:-1,:] += (b_jump_y * beta_eff_y * ystep / h) * ((1-theta_y)/beta_y_k)
    source[ 1:,:] +=  (b_jump_y * beta_eff_y * ystep / h) * (theta_y/beta_y_kp1)
    
    isBoundary_x = np.abs(xstep)
    isBoundary_y = np.abs(ystep)
    
    beta_x += beta_eff_x * isBoundary_x
    beta_x += isOutx2 * (1-isBoundary_x) * beta_p
    beta_x += (1-isOutx2) * (1-isBoundary_x) * beta_m
    beta_y += beta_eff_y * isBoundary_y
    beta_y += isOuty2 * (1-isBoundary_y) * beta_p
    beta_y += (1-isOuty2) * (1-isBoundary_y) * beta_m
    
    beta_sum = beta_x[1:-1,:-1] + beta_x[1:-1,1:] + beta_y[:-1,1:-1] + beta_y[1:,1:-1]
    
    
    
#    source_bcc = np.copy(source)
    
#
#    a_jump_x = a_mesh[:,:-1] * theta_x + a_mesh[:,1:] * (1 - theta_x)
#    a_jump_y = a_mesh[:-1,:] * theta_y + a_mesh[1:,:] * (1 - theta_y)
#    
#    b_jump_x = b_mesh[:,:-1]*theta_x*n1[:,:-1] + b_mesh[:,1:]*(1-theta_x)*n1[:,1:]
#    b_jump_y = b_mesh[:-1,:]*theta_y*n2[:-1,:] + b_mesh[1:,:]*(1-theta_y)*n2[1:,:]
#    
#    #    first additional term
#    source_bcc[:,:-1] += a_jump_x * beta_eff_x*xstep/h**2
#    source_bcc[:,1:] += -a_jump_x * beta_eff_x*xstep/h**2
#    source_bcc[:-1,:] += a_jump_y * beta_eff_y*ystep/h**2
#    source_bcc[1:,:] += -a_jump_y * beta_eff_y*ystep/h**2
#    #    second additional term
#    source_bcc[:,:-1] += (b_jump_x * beta_eff_x * xstep / h) * ((1-theta_x)/beta_x_k)
#    source_bcc[:,1:] +=  (b_jump_x * beta_eff_x * xstep / h) * (theta_x/beta_x_kp1)
#    source_bcc[:-1,:] += (b_jump_y * beta_eff_y * ystep / h) * ((1-theta_y)/beta_y_k)
#    source_bcc[1:,:] +=  (b_jump_y * beta_eff_y * ystep / h) * (theta_y/beta_y_kp1)
    
    
    #record data
    u_prev = u*(1-isOut)
    global iterNum_record
    iterNum_record = 0
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
        del_u = u[1:-1,2:]*beta_x[1:-1,1:] + u[1:-1,0:-2]*beta_x[1:-1,0:-1] +\
                u[2:,1:-1]*beta_y[1:,1:-1] + u[0:-2,1:-1]*beta_y[0:-1,1:-1]
        u_new[1:-1,1:-1] = -h**2/beta_sum * (source[1:-1,1:-1] - del_u/h**2)
      
        u = u_new
        
        # check convergence and print process
        check_convergence_rate = 10**-11
        if(i % int(maxIterNum_/100) < 0.1):
            u_cur = u* (1-isOut)
            maxDif = np.max(np.abs(u_cur - u_prev)) / np.max(np.abs(u_cur))
            L2Dif = hf.L_n_norm(np.abs(u_cur - u_prev)) / hf.L_n_norm(u_cur)
            if(L2Dif < check_convergence_rate):
                break;
            else:
                u_prev = u_cur
            sys.stdout.write("\rProgress: %4g%%" % (i *100.0/maxIterNum_))
            sys.stdout.flush()
    print("")
    return u
setup_grid(101)
r0=0.8
phi = -np.array(hf.XYtoR(xmesh,ymesh) - r0)
plt.matshow(Chi(phi))
plt.colorbar()
plt.matshow(delta(phi,h))
plt.colorbar()
plt.matshow(H(phi,h))
plt.colorbar()
    