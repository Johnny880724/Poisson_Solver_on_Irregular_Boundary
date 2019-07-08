# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:21:33 2019

@author: Johnny Tsao
"""
###
#This is a test code for Laplace equation beta u_xx = 0
#The boundary is a circle r0 = 0.8, u=1, where u jumps from 1 to 0
###

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy import linalg as la

def solution(x,y):
    sol = np.heaviside(-level(x,y),1)
    return sol

def XYtoR(x,y):
    return np.sqrt(x**2+y**2)

def get_theta(u,u_next):
    return np.abs(u)/(np.abs(u)+np.abs(u_next))

def get_effective_beta(beta_p,beta_m,theta):
    ret = beta_p * beta_m / (beta_p * theta + beta_m * (1-theta))
    return ret

##The level defines the boundary
##The boundary here is a circle with radius 0.8
def level(x,y):
    return np.array(XYtoR(x,y) - 0.8)

def grad(f,dx,dy):
    ret_x = np.zeros_like(f)
    ret_y = np.zeros_like(f)
    ret_x[1:-1,:] = (f[:-2,:  ] - f[2:, :])/(2*dx)
    ret_y[:,1:-1] = (f[:  ,:-2] - f[: ,2:])/(2*dy)
    ret_x[0 ,:] = (0 - f[ 1,:])/dx
    ret_x[-1,:] = (f[-1,:] - 0)/dx
    ret_y[: , 0] = (0 - f[ :,1])/dy
    ret_y[: ,-1] = (f[:,-1] - 0)/dy
    return ret_x, ret_y

def div(fx,fy,dx,dy):
    ret_x = np.zeros_like(x)
    ret_y = np.zeros_like(y)
    ret_x[1:-1,:] = (fx[:-2,:  ] - fx[2:, :])/(2*dx)
    ret_y[:,1:-1] = (fy[:  ,:-2] - fy[: ,2:])/(2*dy)
    ret_x[0 ,:] = (0 - fx[ 1,:])/dx
    ret_x[-1,:] = (fx[-1,:] - 0)/dx
    ret_y[: , 0] = (0 - fy[ :,1])/dy
    ret_y[: ,-1] = (fy[:,-1] - 0)/dy
    return ret_x + ret_y

def norm_mesh(x,y):
    ret_l = np.sqrt(x**2 + y**2)
    ret_x = x / ret_l
    ret_y = y / ret_l
    
    return ret_x, ret_y

def L_n_norm(error, n=2):
    error_n = np.power(error, n)
    average_n = np.sum(error_n) / len(error_n.flatten())
    average = np.power(average_n, 1./n)
    return average

def L_n_norm_frame(error,frame, n=2):
    num = np.sum(frame)
    error_n = np.power(error, n) * frame
    average_n = np.sum(error_n) / num
    average = np.power(average_n, 1./n)
    return average

def poisson_solver(iterNum, N_grid = 101, beta_p=1, make_zero = False, plot = False):
    
    a_jump = -1.
    # grid dimension
    grid_min = -1.
    grid_max = 1.
    # grid spacing
    h = (grid_max - grid_min) / (N_grid- 1) 
    
    # define arrays to hold the x and y coordinates
    xy = np.linspace(grid_min,grid_max,N_grid)
    xmesh, ymesh = np.meshgrid(xy,xy)
    
    # solution
    u = np.zeros_like(xmesh)
    
    #level grid
    phi = level(xmesh,ymesh)
    isOut = np.array(np.greater(phi,0.0),dtype = int)
    to_zero = np.ones_like(xmesh)
    if(make_zero):
        to_zero = (1 - isOut)
    
    # the source on the rhs
    #source = np.zeros_like(u)
    source = np.zeros_like(xmesh)
    
    beta_p = beta_p
    beta_m = 1
    beta_x = np.zeros_like(xmesh[1:,:])
    beta_y = np.zeros_like(ymesh[:,1:])
    
    
    isOutx1 = np.array(np.greater(phi[1:,:],0.0),dtype = int)
    isOutx2 = np.array(np.greater(phi[:-1,:],0.0),dtype = int)
    xstep = isOutx1 - isOutx2
#    xstep = np.heaviside(xout,1)
    theta_x = get_theta(phi[:-1,:],phi[1:,:])
    beta_eff_x = get_effective_beta(beta_p,beta_m,theta_x)
    
#    additional term
    source[:-1,:] += a_jump * beta_eff_x*xstep/h**2
    source[1:,:] += -a_jump * beta_eff_x*xstep/h**2
    
    isBoundary_x = np.abs(xstep)
    
    beta_x += beta_eff_x * isBoundary_x
    beta_x += isOutx2 * (1-isBoundary_x) * beta_p
    beta_x += (1-isOutx2) * (1-isBoundary_x) * beta_m
    
    
    isOuty1 = np.array(np.greater(phi[:,1:],0.0),dtype = int)
    isOuty2 = np.array(np.greater(phi[:,:-1],0.0),dtype = int)
    ystep = isOuty1 - isOuty2
#    ystep = np.heaviside(yout,1)
    theta_y = get_theta(phi[:,:-1],phi[:,1:])
    beta_eff_y = get_effective_beta(beta_p,beta_m,theta_y)
    
#    additional term
    source[:,:-1] += a_jump * beta_eff_y*ystep/h**2
    source[:,1:] += -a_jump * beta_eff_y*ystep/h**2
    
    isBoundary_y = np.abs(ystep)
    
    beta_y += beta_eff_y * isBoundary_y
    beta_y += isOuty2 * (1-isBoundary_y) * beta_p
    beta_y += (1-isOuty2) * (1-isBoundary_y) * beta_m
    
    beta_sum = beta_x[:-1,1:-1] + beta_x[1:,1:-1] + beta_y[1:-1,:-1] + beta_y[1:-1,1:]
    
    for i in range(iterNum):
        if(i % int(iterNum/1000) < 0.1):
            sys.stdout.write("\rProgress: %4g%%" % (i *100.0/iterNum))
            sys.stdout.flush()
        # enforce boundary condition
        #  sol = solution1(xmesh, ymesh)
        sol = solution(xmesh, ymesh)
        u[ 0, :] = sol[ 0, :] # x = xmin
        u[-1, :] = sol[-1, :] # x = xmax
        u[ :, 0] = sol[ :, 0] # x = ymin
        u[ :,-1] = sol[ :,-1] # x = ymax
    
        u_new = np.copy(u)
    
        
        
    
        # update u according to Jacobi method formula
        # https://en.wikipedia.org/wiki/Jacobi_method
        del_u = u[2:,1:-1]*beta_x[1:,1:-1] + u[0:-2,1:-1]*beta_x[0:-1,1:-1] +\
                  u[1:-1,2:]*beta_y[1:-1,1:] + u[1:-1,0:-2]*beta_y[1:-1,0:-1]
        u_new[1:-1,1:-1] = -h**2/beta_sum * (source[1:-1,1:-1] - del_u/h**2)
        
        u = u_new * to_zero
    print("")
    #check spectral radius (this is wrong, need (N*N)*(N*N) matrix)
#    D_R = h**2/4. * del_u/h**2
#    eigval = la.eigvals(D_R)
#    plt.matshow(D_R)
#    plt.colorbar()
#    maxeigval = np.max(np.abs(eigval))
#    print(maxeigval)
    
    # error analysis between the solution and analytical solution
    # the error only calculates the area inside the domain
    dif = np.abs(u-solution(xmesh,ymesh))
    L2Dif = L_n_norm_frame(dif,(1-isOut),2)
    maxDif = np.max(dif*(1-isOut))
    print ("N = %d maximum difference after %d iterations was %g" % (N_grid, iterNum, maxDif))
    print ("N = %d L^2 difference after %d iterations was %g" % (N_grid, iterNum, L2Dif))
    if(plot):
        plt.matshow(source)
        #2D color plot of the max difference
        fig_dif = plt.figure("poisson solution max difference %d iterations" % iterNum)
        plt.pcolor(xmesh, ymesh, solution(xmesh, ymesh) - u)
        plt.colorbar()
#        fig_dif.savefig("max_dif_%d.png" % iterNum)
        
        #3D plot of the analytic solution
        fig_an = plt.figure("poisson analytic solution %d iterations" % iterNum)
        ax_an = fig_an.gca(projection='3d')
        surf_an = ax_an.plot_surface(xmesh, ymesh, solution(xmesh, ymesh), cmap=cm.coolwarm)
        fig_an.colorbar(surf_an)
#        fig_an.savefig("analytic_plot_%d.png" % iterNum)
        
        #3D plot of the numerical result
        fig = plt.figure("poisson result %d iterations" % iterNum)
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(xmesh, ymesh, u, cmap=cm.coolwarm)
        fig.colorbar(surf)
#        fig.savefig("sol_plot_%d.png" % iterNum)
        
        plt.show()
    return maxDif, L2Dif

if(__name__ == "__main__"):
    ##generate poisson solver with 10000 iterations
    plt.close("all")
    cur_max, L_2_norm = poisson_solver(10000,101,100,True,True)
#    cur_max, L_2_norm = poisson_solver(10000,101,100,False,True)
    
    #beta test
#    beta_p_array = np.logspace(-2,4,30)
#    cur_max_array=[]
#    L_2_norm_array =[]
#    for beta_p in beta_p_array:
#        print("beta_p = ",beta_p)
#        cur_max, L_2_norm = poisson_solver(10000,101,beta_p,False)
#        cur_max_array.append(cur_max)
#        L_2_norm_array.append(L_2_norm)
#    
#    plt.plot(np.log(beta_p_array),cur_max_array)
#    plt.plot(np.log(beta_p_array),L_2_norm_array)
    
    ##maximum difference and iteration number analysis
#    iterNum = np.logspace(1,4,20,endpoint = True,dtype =int)
#    maxDif_list = []
#    for idx in iterNum:
#        maxDif = poisson_solver(idx)
#        maxDif_list.append(maxDif)
#    fig_max = plt.figure("maximum difference - iteration number")
#    plt.plot(iterNum,maxDif_list)
#    plt.xlabel("iterations")
#    plt.ylabel("maximum difference")
#    fig_max.savefig("max_dif_analysis.png")
    
    ##with / without zero frame 
#    iterNum = np.array((np.arange(1,10) * 5000),dtype = int)
#    maxDif_list = []
#    maxDif_list_frame = []
#    for idx in iterNum:
#        cur_max, L_2_norm = poisson_solver(idx,101,100,False,False)
#        maxDif_list.append(cur_max)
#        cur_max, L_2_norm = poisson_solver(idx,101,100,True,False)
#        maxDif_list_frame.append(cur_max)
#    a1 = plt.figure()
#    plt.plot(iterNum,maxDif_list)
#    a2= plt.figure()
#    plt.plot(iterNum,maxDif_list_frame)
