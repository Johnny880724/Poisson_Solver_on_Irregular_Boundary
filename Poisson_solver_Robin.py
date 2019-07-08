# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:56:57 2019

@author: Johnny Tsao
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import Poisson_solver as ps
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
    
if(__name__ == "__main__"):
    ##generate poisson solver
    plt.close("all")
    
    for i in range(1):
        initialize("u_n")
        u_cur_result = u_init
        u_result = ps.poisson_jacobi_solver(u_cur_result, 200000, (xmesh,ymesh), (beta_p, beta_m),rhs_func, lvl_func, jmp_func)
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
    