# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:02:57 2019

@author: Johnny Tsao
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import Poisson_solver as ps
import poisson_helper_functions as hf

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

def setup_equations(bnd_type_, beta_p_val = 10000):
    
    #offset center
    x0 = 0.6
    y0 = -0.2
    
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
    
    global bnd_type
    bnd_type = bnd_type_
    
    ##below is for testing existing function purposes
    ######################################################################
    def desired_func(x, y):
        return (1 - 0.25 * ((x - x0)**2 + (y - y0)**2))
    
    def solution(x,y):
        sol = np.heaviside(-lvl_func(x,y),1)*(desired_func(x,y))
        return sol
    global sol_func
    sol_func = solution
    
    
    
    ##boundary conditions / jump conditions
    def jump_condition(x,y,u):
        u_desired = desired_func(x,y)
        u_n_desired = beta_m * hf.norm_grad(u_desired,(x,y),lvl_func)
        #for Robin boundary condition only
        def sigma(x,y):
            return -0.25 * (hf.XYtoR(x-x0, y-y0)+10**-17)
        
        sigma_Robin = sigma(x, y)
        g_Robin = u_desired + sigma_Robin * u_n_desired
        #################################
        
        if(bnd_type == "Dirichlet"):
            a_mesh = - u_desired
            b_mesh = - beta_m * hf.grad_frame(u,(x,y),lvl_func)
            
        elif(bnd_type == "Neumann"):
            a_mesh = - u
            b_mesh = - beta_m * u_n_desired
        
        elif(bnd_type == "Robin_u"):
            a_mesh = - u
            b_mesh = - beta_m * (g_Robin - u)/sigma_Robin
            
        elif(bnd_type == "Robin_u_n"):
            a_mesh = -(g_Robin - sigma_Robin*u_n_desired)
            b_mesh = - beta_m * hf.grad_frame(u,(x,y),lvl_func)
            
        elif(bnd_type == "exact"):
            a_mesh = - u_desired
            b_mesh = - beta_m * u_n_desired
        else:
            raise Exception("error: invalid boundary format")
        return (a_mesh, b_mesh)

    global jmp_func
    jmp_func = jump_condition
    ######################################################################
    
if(__name__ == "__main__"):
    ##generate poisson solver
    plt.close("all")
    setup_grid(33)
    u_cur_result = u_init
    for i in range(1000):
        setup_equations("Neumann")
        u_result = ps.poisson_jacobi_solver(u_cur_result, 200000, (xmesh,ymesh), (beta_p, beta_m),rhs_func, lvl_func, jmp_func)
        u_n_result = hf.grad_frame(u_result, (xmesh, ymesh), lvl_func)
#        plt.matshow(u_n_result)
#        plt.colorbar()
        fig_label = i
#        hf.plot3d_all(u_result, (xmesh, ymesh), sol_func,fig_label,[False,False,False,False])
        hf.print_error(u_result, (xmesh, ymesh), sol_func)
        u_cur_result = u_result
        u_n_result = hf.grad_frame(u_result,(xmesh,ymesh),lvl_func)
        u_n_anal = hf.grad_frame(sol_func(xmesh, ymesh),(xmesh,ymesh),lvl_func)
        print(hf.L_n_norm(np.abs(u_n_result - u_n_anal),2))
    hf.plot3d_all(u_result, (xmesh, ymesh), sol_func,fig_label,[False,False,False,False])