# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:13:57 2019

@author: Johnny Tsao
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy import linalg as la


r0 = 0.7878973648236
def rhs(x,y):
    return -np.ones_like(x)

def solution(x,y):
    sol = np.heaviside(-level(x,y),1)*(1 + 0.25*(r0**2 - XYtoR(x,y)**2))
    return sol

def XYtoR(x,y):
    return np.sqrt(x**2+y**2)

def get_theta(u,u_next):
    return np.abs(u)/(np.abs(u)+np.abs(u_next))

def get_effective_beta(beta_p,beta_m,theta):
    ret = beta_p * beta_m / (beta_p * theta + beta_m * (1-theta))
    return ret

def level(x,y):
    return np.array(XYtoR(x,y) - r0)

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
    
    #level grid
    phi = level(xmesh,ymesh)
    
    # normal
    r_temp = XYtoR(xmesh, ymesh) + 10**-13
    n1, n2 = xmesh/r_temp, ymesh/r_temp
    
    # the source on the rhs
    #source = np.zeros_like(u)
    source = np.zeros_like(xmesh)
    
    isOut = np.array(np.greater(phi,0.0),dtype = int)
    source += (1.0 - isOut) * rhs(xmesh, ymesh)
    
    
    beta_p = beta_p
    beta_m = 1.
    beta_x = np.zeros_like(xmesh[:,1:])
    beta_y = np.zeros_like(ymesh[1:,:])
    
#    a_jump = a+ - a-
    a_mesh = -np.ones_like(xmesh)
#    b_jump = beta+ * u_n+ - beta- * u_n-
    b_mesh = -beta_m * -(XYtoR((xmesh),(ymesh)))
    
    
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
    
    beta_eff_x += xstep_p * get_effective_beta(beta_p,beta_m,theta_x)
    beta_eff_x += xstep_m * get_effective_beta(beta_m,beta_p,theta_x)
    beta_eff_y += ystep_p * get_effective_beta(beta_p,beta_m,theta_y)
    beta_eff_y += ystep_m * get_effective_beta(beta_m,beta_p,theta_y)
    
    a_jump_x = a_mesh[:,:-1] * (1-theta_x) + a_mesh[:,1:] * theta_x
    a_jump_y = a_mesh[:-1,:] * (1-theta_y) + a_mesh[1:,:] * theta_y
    
    b_jump_x = b_mesh[:,:-1]*(1-theta_x)*n1[:,:-1] + b_mesh[:,1:]*theta_x*n1[:,1:]
    b_jump_y = b_mesh[:-1,:]*(1-theta_y)*n2[:-1,:] + b_mesh[1:,:]*theta_y*n2[1:,:]
#    first additional term
    source[:,:-1] += a_jump_x * beta_eff_x*xstep/h**2
    source[:, 1:] += -a_jump_x * beta_eff_x*xstep/h**2
    source[:-1,:] += a_jump_y * beta_eff_y*ystep/h**2
    source[ 1:,:] += -a_jump_y * beta_eff_y*ystep/h**2
    
#    theta_x_k   = (1. + (1. - 2*theta_x)*xstep)/2.
#    theta_x_kp1 = (1. - (1. - 2*theta_x)*xstep)/2.
#    theta_y_k   = (1. + (1. - 2*theta_y)*ystep)/2.
#    theta_y_kp1 = (1. - (1. - 2*theta_y)*ystep)/2.
    beta_x_k   = ((beta_p + beta_m) + (beta_p - beta_m)*xstep)/2
    beta_x_kp1 = ((beta_p + beta_m) - (beta_p - beta_m)*xstep)/2
    beta_y_k   = ((beta_p + beta_m) + (beta_p - beta_m)*ystep)/2
    beta_y_kp1 = ((beta_p + beta_m) - (beta_p - beta_m)*ystep)/2
    
#    second additional term
#    source[:-1,:] += (b_jump_x*n1[:-1,:] * beta_eff_x * xstep / h) * (theta_x/beta_x_k)
#    source[1:,:] +=  (b_jump_x*n1[ 1:,:] * beta_eff_x * xstep / h) * ((1-theta_x)/beta_x_kp1)
#    source[:,:-1] += (b_jump_y*n2[:,:-1] * beta_eff_y * ystep / h) * (theta_y/beta_y_k)
#    source[:,1:] +=  (b_jump_y*n2[:, 1:] * beta_eff_y * ystep / h) * ((1-theta_y)/beta_y_kp1)
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
    
    
    #record data
    iterNum_array = []
    maxDif_array = []
    L2Dif_array = []
    for i in range(iterNum):
        if(i % int(iterNum/100) < 0.1):
            dif = np.abs(u-solution(xmesh,ymesh))
            L2Dif = L_n_norm_frame(dif,(1-isOut),2)
            maxDif = np.max(dif*(1-isOut))
            iterNum_array.append(i)
            maxDif_array.append(maxDif)
            L2Dif_array.append(L2Dif)
            sys.stdout.write("\rProgress: %4g%%" % (i *100.0/iterNum))
            sys.stdout.flush()
        # enforce boundary condition
        #  sol = solution1(xmesh, ymesh)
        sol = solution(xmesh, ymesh)
        u[ 0, :] = sol[ 0, :] # y = y top
        u[-1, :] = sol[-1, :] # y = y bottom
        u[ :, 0] = sol[ :, 0] # x = x left
        u[ :,-1] = sol[ :,-1] # x = x right
    
        u_new = np.copy(u)
    
        
        
    
        # update u according to Jacobi method formula
        # https://en.wikipedia.org/wiki/Jacobi_method
        del_u = u[1:-1,2:]*beta_x[1:-1,1:] + u[1:-1,0:-2]*beta_x[1:-1,0:-1] +\
                u[2:,1:-1]*beta_y[1:,1:-1] + u[0:-2,1:-1]*beta_y[0:-1,1:-1]
        u_new[1:-1,1:-1] = -h**2/beta_sum * (source[1:-1,1:-1] - del_u/h**2)
      
        u = u_new * (1-isOut)
    print("")
    
    # error analysis between the solution and analytical solution
    dif = np.abs(u-solution(xmesh,ymesh))
    L2Dif = L_n_norm_frame(dif,(1-isOut),2)
    maxDif = np.max(dif*(1-isOut))
    print ("N = %d maximum difference after %d iterations was %g" % (N_grid, iterNum, maxDif))
    print ("N = %d L^2 difference after %d iterations was %g" % (N_grid, iterNum, L2Dif))
    if(plot):
#        plt.matshow(phi)
#        plt.colorbar()
#        plt.matshow(theta_x*(1-isOutx2))
#        plt.colorbar()
#        plt.matshow(theta_y*(1-isOuty2))
#        plt.colorbar()
#        #2D color plot of the max difference
        fig_dif = plt.figure("poisson solution max difference %d iterations" % iterNum)
        plt.title("poisson solver error")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.pcolor(xmesh, ymesh, (solution(xmesh, ymesh) - u)/(solution(xmesh, ymesh)+1.0e-17))
        plt.colorbar()
#        fig_dif.savefig("max_dif_%d.png" % iterNum)
        
#        #2D color plot of the max difference
#        fig_dif = plt.figure("poisson solution max difference %d iterations" % iterNum)
#        plt.title("poisson solver analytical solution")
#        plt.xlabel("x")
#        plt.ylabel("y")
#        plt.pcolor(xmesh, ymesh, solution(xmesh, ymesh))
#        plt.colorbar()
        
        #2D color plot of the max difference
#        fig_dif = plt.figure("poisson solution max difference %d iterations" % iterNum)
#        plt.title("poisson solver result")
#        plt.xlabel("x")
#        plt.ylabel("y")
#        plt.pcolor(xmesh, ymesh,u)
#        plt.colorbar()
        
        #3D plot of the analytic solution
        fig_an = plt.figure("poisson analytic solution %d iterations" % iterNum)
        ax_an = fig_an.gca(projection='3d')
        ax_an.set_title("analytic solution")
        surf_an = ax_an.plot_surface(xmesh, ymesh, solution(xmesh, ymesh), cmap=cm.coolwarm)
        fig_an.colorbar(surf_an)
#        fig_an.savefig("analytic_plot_%d.png" % iterNum)
        
        #3D plot of the numerical result
        fig = plt.figure("poisson result %d iterations" % iterNum)
        ax = fig.gca(projection='3d')
        ax_an.set_title("result")
        surf = ax.plot_surface(xmesh, ymesh, u, cmap=cm.coolwarm)
        fig.colorbar(surf)
#        fig.savefig("sol_plot_%d.png" % iterNum)
        
        #3D plot of the error
        fig_an = plt.figure("poisson error %d iterations" % iterNum)
        ax_an = fig_an.gca(projection='3d')
        ax_an.set_title("error")
        surf_an = ax_an.plot_surface(xmesh, ymesh, ((u-solution(xmesh, ymesh))/(solution(xmesh, ymesh)+1.0e-17))*(1-isOut), cmap=cm.coolwarm)
        fig_an.colorbar(surf_an)
        
        plt.show()
    return iterNum_array, maxDif_array, L2Dif_array

if(__name__ == "__main__"):
    ##generate poisson solver with 10000 iterations
    plt.close("all")
#    iterNum_array = [51,101,151,201,251,301]
#    iterNum_array = [21,41,81,161,321]
    iterNum_array = [101]
    plot_iterNum = []
    plot_L2Dif = []
    plot_maxDif = []
    for idx in iterNum_array:
        iterNum, maxDif, L2Dif = poisson_solver(2*(idx-1)**2,idx,10000,True)
        plot_iterNum.append(iterNum)
        plot_L2Dif.append(L2Dif)
        plot_maxDif.append(maxDif)
    plt.figure()
    for i in range(len(iterNum_array)):
        plt.plot(plot_iterNum[i],np.log(plot_L2Dif[i]))
    


