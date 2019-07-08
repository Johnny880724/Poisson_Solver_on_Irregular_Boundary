# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:21:47 2019

@author: Johnny Tsao
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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

def norm_grad(u_, mesh_, lvl_func_):
    xmesh, ymesh = mesh_
    phi = lvl_func_(xmesh,ymesh)
    n1, n2 = grad(phi,xmesh[0,1]-xmesh[0,0],ymesh[1,0]-ymesh[0,0])
    n_sum = XYtoR(n1, n2) + 10**-17
    n1_norm = n1/n_sum
    n2_norm = n2/n_sum
    u_nx,u_ny = grad(u_, xmesh[0,1]-xmesh[0,0],ymesh[1,0]-ymesh[0,0])
    u_n = u_nx * n1_norm + u_ny * n2_norm
    
    return u_n

def div(fx,fy,dx,dy):
    ret_x = np.zeros_like(fx)
    ret_y = np.zeros_like(fy)
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

def grad_frame(u_, mesh_, lvl_func_):
    xmesh, ymesh = mesh_
    phi = lvl_func_(xmesh,ymesh)
    isOut = np.array(np.greater(phi,0.0),dtype = int)
    isOutx1 = np.array(np.greater(phi[1:,:],0.0),dtype = int)
    isOutx2 = np.array(np.greater(phi[:-1,:],0.0),dtype = int)
    isOuty1 = np.array(np.greater(phi[:,1:],0.0),dtype = int)
    isOuty2 = np.array(np.greater(phi[:,:-1],0.0),dtype = int)
    # normal
    
    n1, n2 = grad(phi,xmesh[0,1]-xmesh[0,0],ymesh[1,0]-ymesh[0,0])
    n_sum = XYtoR(n1, n2) + 10**-17
    n1_norm = n1/n_sum
    n2_norm = n2/n_sum
#    step is 1 if k+1 is out and k is in
#    step is -1 if k is out and k+1 is in
#    step is 0 if both out or in
    xstep = isOutx1 - isOutx2
    ystep = isOuty1 - isOuty2
    xstep_p = np.array(np.greater( xstep,0.0),dtype = int)
    xstep_m = np.array(np.greater(-xstep,0.0),dtype = int)
    
    ystep_p = np.array(np.greater( ystep,0.0),dtype = int)
    ystep_m = np.array(np.greater(-ystep,0.0),dtype = int)
    
    u_ghost_x = np.copy(u_) * (1-isOut)
    u_ghost_y = np.copy(u_) * (1-isOut)
    
    u_ghost_x[:-2,:] += -u_[ 2:,:]*xstep_m[:-1,:] + 2*u_[ 1:-1,:]*xstep_m[:-1,:]
    u_ghost_x[ 2:,:] += -u_[:-2,:]*xstep_p[ 1:,:] + 2*u_[ 1:-1,:]*xstep_p[ 1:,:]
    u_ghost_y[:,:-2] += -u_[:, 2:]*ystep_m[:,:-1] + 2*u_[:, 1:-1]*ystep_m[:,:-1]
    u_ghost_y[:, 2:] += -u_[:,:-2]*ystep_p[:, 1:] + 2*u_[:, 1:-1]*ystep_p[:, 1:]
    
    u_nx,temp = grad(u_ghost_y,xmesh[0,1]-xmesh[0,0],ymesh[1,0]-ymesh[0,0])
    temp,u_ny = grad(u_ghost_x,xmesh[0,1]-xmesh[0,0],ymesh[1,0]-ymesh[0,0])
    u_n = u_nx * n1_norm + u_ny * n2_norm
    
    return u_n * (1-isOut)

def print_error(u_result_, mesh_, sol_func_):
    xmesh,ymesh = mesh_
    dif = np.abs(u_result_ - sol_func_(xmesh,ymesh))
    print(dif)
    L2Dif = L_n_norm(dif,2)
    maxDif = np.max(dif)
    print("maximum error: %f" % maxDif)
    print("L2 error: %f" % L2Dif)

def plot3d_all(u_result_, mesh_, sol_func_,fig_label_, toPlot_ = [True, True, True, True]):
    xmesh, ymesh = mesh_
    sol_mesh = sol_func_(xmesh, ymesh)
    if(toPlot_[0]):
        #2D color plot of the max difference
        fig_er = plt.figure("poisson solution error %d" % fig_label_)
        plt.pcolor(xmesh, ymesh, (sol_mesh - u_result_)/(sol_mesh+1.0e-17))
        plt.colorbar()
#        fig_dif.savefig("max_dif_%d.png" % iterNum)
    
    
    plot_max = np.max(sol_mesh)
    plot_min = np.min(sol_mesh)
    if(toPlot_[1]):
        #3D plot of the analytic solution
        fig_an = plt.figure("poisson analytic solution %d" % fig_label_)
        ax_an = fig_an.gca(projection='3d')
        surf_an = ax_an.plot_surface(xmesh, ymesh, sol_mesh, cmap=cm.coolwarm)
        fig_an.colorbar(surf_an)
        ax_an.set_zlim3d(plot_min, plot_max)
#        fig_an.savefig("analytic_plot_%d.png" % iterNum)
    
    if(toPlot_[2]):
        #3D plot of the numerical result
        fig = plt.figure("poisson result %d" % fig_label_)
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(xmesh, ymesh, u_result_, cmap=cm.coolwarm)
        fig.colorbar(surf)
        ax.set_zlim3d(plot_min, plot_max)
#        fig.savefig("sol_plot_%d.png" % iterNum)
    if(toPlot_[3]):
        #3D plot of the error
        fig_dif = plt.figure("poisson difference %d" % fig_label_)
        ax_dif = fig_dif.gca(projection='3d')
        surf_dif = ax_dif.plot_surface(xmesh, ymesh, u_result_ - sol_mesh, cmap=cm.coolwarm)
        fig_dif.colorbar(surf_dif)
#        ax_dif.set_zlim3d(-0.2,0.2)
    
    plt.show()

if(__name__ == "__main__"):
    print("Poisson solver helper function file")
