# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 16:09:56 2019

@author: Johnny Tsao
"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy import linalg as la


def XYtoR(x,y):
    return np.sqrt(x**2+y**2)

def grad(f,dx,dy):
    ret_x = np.zeros_like(f)
    ret_y = np.zeros_like(f)
    ret_y[1:-1,:] = (f[2:,:  ] - f[:-2, :])/(2*dx)
    ret_x[:,1:-1] = (f[:  ,2:] - f[: ,:-2])/(2*dy)
    ret_y[0 ,:] = (f[ 1,:] - 0)/dx
    ret_y[-1,:] = (0 - f[-1,:])/dx
    ret_x[: , 0] = (f[ :,1] - 0)/dy
    ret_x[: ,-1] = (0 - f[:,-1])/dy
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
    
#rhs solution
r0 = 0.765467899876
def rhs(x,y):
    return -np.ones_like(x)
x0 = -0.3
y0 = 0.5
    
def level(x,y):
    return np.array(XYtoR(x,y) - r0)
    
def desired_func(x, y):
    return (1 - 0.25 * ((x - x0)**2 + (y - y0)**2))
    
def solution(x,y):
    sol = np.heaviside(-level(x,y),1)*(desired_func(x,y))
    return sol
    global sol_func
    sol_func = solution
    
#Robin boundary
def m(x,y):
    return 0.5 * (XYtoR(x,y) + 10**-17)

def n(x,y):
    return np.ones_like(x)

def g(x,y):
    return np.zeros_like(x)

def get_u_from_un(u_n,x,y):
    return (g(x,y) - n(x,y)*u_n)/m(x,y)

def get_un_from_u(u,x,y):
    return (g(x,y) - m(x,y)*u)/n(x,y)
    

def poisson_solver(iterNum, N_grid = 101, beta_p=1, plot = False):
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
    
    # ghost points for gradient
    u_ghost_x = np.zeros_like(u)
    u_ghost_y = np.zeros_like(u)
    
    #level grid
    phi = level(xmesh,ymesh)
    
    # normal
    r_temp = XYtoR(xmesh, ymesh)  + 10**-17
    n1, n2 = xmesh/r_temp, ymesh/r_temp
    
    # the source on the rhs
    #source = np.zeros_like(u)
    source = np.zeros_like(xmesh)
    
    isOut = np.array(np.greater(phi,0.0),dtype = int)
    source += (1.0 - isOut) * rhs(xmesh, ymesh)
    
    
    beta_p = beta_p
    beta_m = 1.
    beta_x = np.zeros_like(xmesh[1:,:])
    beta_y = np.zeros_like(ymesh[:,1:])
    
#    a_jump = a+ - a-
    a_jump_init = -1
    a_mesh_init = a_jump_init * np.ones_like(xmesh)
#    b_jump = beta+ * u_n+ - beta- * u_n-
    b_jump_init = 0.5 * beta_m * r0
    b_mesh_init = b_jump_init * np.ones_like(xmesh)
    
    
    isOutx1 = np.array(np.greater(phi[1:,:],0.0),dtype = int)
    isOutx2 = np.array(np.greater(phi[:-1,:],0.0),dtype = int)
    isOuty1 = np.array(np.greater(phi[:,1:],0.0),dtype = int)
    isOuty2 = np.array(np.greater(phi[:,:-1],0.0),dtype = int)
    
#    step is 1 if k+1 is out and k is in
#    step is -1 if k is out and k+1 is in
#    step is 0 if both out or in
    xstep = isOutx1 - isOutx2
    ystep = isOuty1 - isOuty2
    
    theta_x = get_theta(phi[:-1,:],phi[1:,:])
    theta_y = get_theta(phi[:,:-1],phi[:,1:])
    
    beta_eff_x = get_effective_beta(beta_p,beta_m,theta_x)
    beta_eff_y = get_effective_beta(beta_p,beta_m,theta_y)
    
#    initializing jump conditions
    a_mesh = a_mesh_init
    b_mesh = b_mesh_init
    
#    theta_x_k   = (1. + (1. - 2*theta_x)*xstep)/2.
#    theta_x_kp1 = (1. - (1. - 2*theta_x)*xstep)/2.
#    theta_y_k   = (1. + (1. - 2*theta_y)*ystep)/2.
#    theta_y_kp1 = (1. - (1. - 2*theta_y)*ystep)/2.
    beta_x_k   = ((beta_p + beta_m) + (beta_p - beta_m)*xstep)/2
    beta_x_kp1 = ((beta_p + beta_m) - (beta_p - beta_m)*xstep)/2
    beta_y_k   = ((beta_p + beta_m) + (beta_p - beta_m)*ystep)/2
    beta_y_kp1 = ((beta_p + beta_m) - (beta_p - beta_m)*ystep)/2
    

    
    isBoundary_x = np.abs(xstep)
    isBoundary_y = np.abs(ystep)
    
    beta_x += beta_eff_x * isBoundary_x
    beta_x += isOutx2 * (1-isBoundary_x) * beta_p
    beta_x += (1-isOutx2) * (1-isBoundary_x) * beta_m
    beta_y += beta_eff_y * isBoundary_y
    beta_y += isOuty2 * (1-isBoundary_y) * beta_p
    beta_y += (1-isOuty2) * (1-isBoundary_y) * beta_m
    
    beta_sum = beta_x[:-1,1:-1] + beta_x[1:,1:-1] + beta_y[1:-1,:-1] + beta_y[1:-1,1:]
    
    xstep_p = np.array(np.greater( xstep,0.0),dtype = int)
    xstep_m = np.array(np.greater(-xstep,0.0),dtype = int)
    
    ystep_p = np.array(np.greater( ystep,0.0),dtype = int)
    ystep_m = np.array(np.greater(-ystep,0.0),dtype = int)
    
    
    #record data
    iterNum_array = []
    maxDif_array = []
    L2Dif_array = []
    for i in range(iterNum):
        source_bcc = np.copy(source)
    
        a_jump_x = a_mesh[:-1,:] * theta_x + a_mesh[1:,:] * (1 - theta_x)
        a_jump_y = a_mesh[:,:-1] * theta_y + a_mesh[:,1:] * (1 - theta_y)
        
        b_jump_x = b_mesh[:-1,:]*theta_x*n1[:-1,:] + b_mesh[1:,:]*(1-theta_x)*n1[1:,:]
        b_jump_y = b_mesh[:,:-1]*theta_y*n2[:,:-1] + b_mesh[:,1:]*(1-theta_y)*n2[:,1:]
    #    first additional term
        source_bcc[:-1,:] += a_jump_x * beta_eff_x*xstep/h**2
        source_bcc[1:,:] += -a_jump_x * beta_eff_x*xstep/h**2
        source_bcc[:,:-1] += a_jump_y * beta_eff_y*ystep/h**2
        source_bcc[:,1:] += -a_jump_y * beta_eff_y*ystep/h**2
        #    second additional term
        source_bcc[:-1,:] += (b_jump_x * beta_eff_x * xstep / h) * ((1-theta_x)/beta_x_k)
        source_bcc[1:,:] +=  (b_jump_x * beta_eff_x * xstep / h) * (theta_x/beta_x_kp1)
        source_bcc[:,:-1] += (b_jump_y * beta_eff_y * ystep / h) * ((1-theta_y)/beta_y_k)
        source_bcc[:,1:] +=  (b_jump_y * beta_eff_y * ystep / h) * (theta_y/beta_y_kp1)
        
        if(i % int(iterNum/100) < 0.1):
            dif = np.abs(u-solution(xmesh,ymesh))
            L2Dif = L_n_norm_frame(dif,(1-isOut),2)
            maxDif = np.max(dif*(1-isOut))
            iterNum_array.append(i)
            maxDif_array.append(maxDif)
            L2Dif_array.append(L2Dif)
            sys.stdout.write("\rProgress: %4g%%" % (i *100.0/iterNum))
            sys.stdout.flush()
            
            #formulism 2
#            u_ghost_x = np.copy(u) * (1-isOut)
#            u_ghost_y = np.copy(u) * (1-isOut)
#            
#            u_ghost_x[:-2,:] += -u[ 2:,:]*xstep_m[:-1,:] + 2*u[ 1:-1,:]*xstep_m[:-1,:]
#            u_ghost_x[ 2:,:] += -u[:-2,:]*xstep_p[ 1:,:] + 2*u[ 1:-1,:]*xstep_p[ 1:,:]
#            u_ghost_y[:,:-2] += -u[:, 2:]*ystep_m[:,:-1] + 2*u[:, 1:-1]*ystep_m[:,:-1]
#            u_ghost_y[:, 2:] += -u[:,:-2]*ystep_p[:, 1:] + 2*u[:, 1:-1]*ystep_p[:, 1:]
#            
#            u_nx,temp = grad(u_ghost_y,h,h)
#            temp,u_ny = grad(u_ghost_x,h,h)
#            u_n = u_nx * n1 + u_ny * n2
#            
#            b_mesh = -beta_m * u_n[:,:]
#            a_mesh = -(get_u_from_un(u_n, xmesh, ymesh))
            #formulism 2
            a_mesh = - u[:,:]
            b_mesh = -(get_u_from_un(u, xmesh, ymesh))
            
            
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
        u_new[1:-1,1:-1] = -h**2/beta_sum * (source_bcc[1:-1,1:-1] - del_u/h**2)
      
        u = u_new
        
#    u = solution(xmesh,ymesh)
    u_ghost_x = np.copy(u) * (1-isOut)
    u_ghost_y = np.copy(u) * (1-isOut)
    
    u_ghost_x[:-2,:] += -u[ 2:,:]*xstep_m[:-1,:] + 2*u[ 1:-1,:]*xstep_m[:-1,:]
    u_ghost_x[ 2:,:] += -u[:-2,:]*xstep_p[ 1:,:] + 2*u[ 1:-1,:]*xstep_p[ 1:,:]
    u_ghost_y[:,:-2] += -u[:, 2:]*ystep_m[:,:-1] + 2*u[:, 1:-1]*ystep_m[:,:-1]
    u_ghost_y[:, 2:] += -u[:,:-2]*ystep_p[:, 1:] + 2*u[:, 1:-1]*ystep_p[:, 1:]
    
    u_nx,temp = grad(u_ghost_y,h,h)
    temp,u_ny = grad(u_ghost_x,h,h)
    u_n = u_nx * n1 + u_ny * n2
    print("")
    plt.matshow(u_n*(1-isOut))
    plt.colorbar()
    plt.matshow(b_mesh*(1-isOut))
    plt.colorbar()
    
    # error analysis between the solution and analytical solution
    dif = np.abs(u-solution(xmesh,ymesh))
    L2Dif = L_n_norm_frame(dif,(1-isOut),2)
    maxDif = np.max(dif*(1-isOut))
    print ("N = %d maximum difference after %d iterations was %g" % (N_grid, iterNum, maxDif))
    print ("N = %d L^2 difference after %d iterations was %g" % (N_grid, iterNum, L2Dif))
    if(plot):
        #2D color plot of the source
        fig_source = plt.figure("source")
        plt.pcolor(xmesh, ymesh, source*(1-isOut))
        plt.colorbar()
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
        
        #3D plot of the error
        fig_an = plt.figure("poisson error %d iterations" % iterNum)
        ax_an = fig_an.gca(projection='3d')
        surf_an = ax_an.plot_surface(xmesh, ymesh, u - solution(xmesh, ymesh), cmap=cm.coolwarm)
        fig_an.colorbar(surf_an)
        
        plt.show()
    return iterNum_array, maxDif_array, L2Dif_array


if(__name__ == "__main__"):
    ##generate poisson solver with 10000 iterations
    plt.close("all")
#    iterNum_array = [51,101,151,201,251,301]
    iterNum_array = [101]
    plot_iterNum = []
    plot_L2Dif = []
    plot_maxDif = []
    for idx in iterNum_array:
        iterNum, maxDif, L2Dif = poisson_solver(10000,idx,10000,True)
        plot_iterNum.append(iterNum)
        plot_L2Dif.append(L2Dif)
        plot_maxDif.append(maxDif)
    plt.figure()
    for i in range(len(iterNum_array)):
        plt.plot(plot_iterNum[i],np.log(plot_L2Dif[i]))