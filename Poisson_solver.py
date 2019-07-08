# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 12:07:12 2019

@author: Johnny Tsao
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy import linalg as la

import poisson_helper_functions as hf

def initialize(jump,N_grid_val = 101, beta_p_val = 10000):
    
    #offset center
    x0 = 0.0
    y0 = 0.0
    
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
    
    #Poisson equation coefficients
    global beta_p
    beta_p = beta_p_val
    global beta_m
    beta_m = 1.
    
    
    #rhs solution
    global r0
    r0 = 0.765467899876
    def rhs(x,y):
        return -np.ones_like(x)
    
    global rhs_func
    rhs_func = rhs
    
    
    def level(x,y):
        return np.array(hf.XYtoR(x,y) - r0)
    
    global lvl_func
    lvl_func = level
    
    def desired_func(x, y):
        return (1 - 0.25 * ((x - x0)**2 + (y - y0)**2))
    
    def solution(x,y):
        sol = np.heaviside(-lvl_func(x,y),1)*(desired_func(x,y))
        return sol
    global sol_func
    sol_func = solution
    
    ##boundary conditions / jump conditions
    def jump_condition(x,y,u_or_un):
#        ##Dirichlet boundary
#        #    a_jump = a+ - a-
#        a_mesh = - desired_func(x,y)
#        
#        ##Neuman boundary
#        #    b_jump = beta+ * u_n+ - beta- * u_n-
#        b_mesh = beta_m * (-0.5) * (XYtoR((x - x0),(y - y0)))
#        return (a_mesh, b_mesh)
        ##Robin boundary
        
        def m(x,y):
            return np.ones_like(x)
        
        def n(x,y):
            return -0.25 * (hf.XYtoR(x-x0, y-y0)+10**-17)/beta_m
        
        def g(x,y):
            return np.ones_like(x)
        
        def get_u_from_un(u_n,x,y):
            return (g(x,y) - n(x,y)*u_n)/m(x,y)
        
        def get_un_from_u(u,x,y):
            return (g(x,y) - m(x,y)*u)/n(x,y)
        
        if(jump == "u_n"):
            a_mesh = -get_u_from_un(u_or_un,x,y)
            b_mesh = -beta_m * u_or_un
        elif (jump == "u"):
            b_mesh = -get_un_from_u(u_or_un,x,y)
            a_mesh = -u_or_un
        else:
            print("error!!")
        return (a_mesh, b_mesh)

    global jmp_func
    jmp_func = jump_condition
    
def poisson_jacobi_solver(u_init_, maxIterNum_, mesh_, beta_, rhs_func_, lvl_func_, jmp_func_):
    xmesh, ymesh = mesh_
    beta_p, beta_m = beta_
    #level grid
    phi = lvl_func_(xmesh,ymesh)
    h = xmesh[0,1]-xmesh[0,0]
    u = u_init_
    
    # normal
    r_temp = hf.XYtoR(xmesh, ymesh) + 10**-17
    n1, n2 = xmesh/r_temp, ymesh/r_temp
    
    # the source on the rhs
    #source = np.zeros_like(u)
    source = np.zeros_like(xmesh)
    
    isOut = np.array(np.greater(phi,0.0),dtype = int)
    source += (1.0 - isOut) * rhs_func_(xmesh, ymesh)
    
    a_mesh, b_mesh = jmp_func_(xmesh, ymesh, hf.grad_frame(u_init_, (xmesh, ymesh), lvl_func_))
    
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
        check_convergence_rate = 10**-7
        if(i % int(maxIterNum_/100) < 0.1):
            u_cur = u* (1-isOut)
            maxDif = np.max(np.abs(u_cur - u_prev)) / np.max(np.abs(u_cur))
            if(maxDif < check_convergence_rate):
                break;
            else:
                u_prev = u_cur
            sys.stdout.write("\rProgress: %4g%%" % (i *100.0/maxIterNum_))
            sys.stdout.flush()
            
    return u

if(__name__ == "__main__"):
    ##generate poisson solver
    plt.close("all")
    
    for i in range(1):
        initialize("u_n")
        u_cur_result = u_init
        u_result = poisson_jacobi_solver(u_cur_result, 200000, (xmesh,ymesh), (beta_p, beta_m),rhs_func, lvl_func, jmp_func)
        u_n_result = hf.grad_frame(u_result, (xmesh, ymesh), lvl_func)
#        plt.matshow(u_n_result)
#        plt.colorbar()
        fig_label = i
        hf.plot3d_all(u_result, (xmesh, ymesh), sol_func,fig_label,[False,True,True,True])
        u_cur_result = u_result
#    initialize()
#    u_result2 = poisson_jacobi_solver(u_result1, 100000, (xmesh,ymesh), (beta_p, beta_m),rhs_func, lvl_func, jmp_func)
#    u_n_result2 = grad_frame(u_result2, (xmesh, ymesh), lvl_func)
#    plt.matshow(u_n_result2)
#    plt.colorbar()
#    plot3d_all(u_result2, (xmesh, ymesh), sol_func,[False, False,False,True])
#    b_mesh = -beta_m * u_n[:,:]
#    a_mesh = -(get_u_from_un(u_n, xmesh, ymesh))
##    formulism 2
#    a_mesh = - u[:,:]
#    b_mesh = -(get_u_from_un(u, xmesh, ymesh))
else:
    print("Poisson solver imported")

        
    
    