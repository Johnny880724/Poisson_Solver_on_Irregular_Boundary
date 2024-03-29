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

def XYtoTheta(x,y):
    return np.arctan2(y,x)

def grad(f,dx,dy):
#    ret_x = np.zeros_like(f)
#    ret_y = np.zeros_like(f)
#    ret_y[1:-1, :] = (f[:-2, :] - f[ 2:, :])/(2*dx)
#    ret_x[ :,1:-1] = (f[ :,:-2] - f[ :, 2:])/(2*dy)
#    ret_y[0 , :] = (f[ 1, :] - f[ 0, :])/dx
#    ret_y[-1, :] = (f[-1, :] - f[-2, :])/dx
#    ret_x[ :, 0] = (f[ :, 1] - f[ :, 0])/dy
#    ret_x[ :,-1] = (f[ :,-1] - f[ :,-2])/dy
    ret_y, ret_x = np.gradient(f,dx,dy)
    return ret_x, ret_y

def abs_grad(f,dx,dy):
    grad_y, grad_x = np.gradient(f,dx,dy)
    return np.sqrt(grad_x**2 + grad_y**2)

#def laplacian(f,dx,dy):
#    grad_x, grad_y = grad(f,dx,dy)
#    return div(grad_x,grad_y,dx,dy)

def laplace(f,dx,dy):
    ret = np.zeros_like(f)
    ret[1:-1,1:-1] = (f[1:-1,2:] + f[1:-1,0:-2] + f[2:,1:-1] + f[0:-2,1:-1] - 4*f[1:-1,1:-1])/dx**2
#    ret[1:-1,0] = (f[1:-1,2:] + f[1:-1,0:-2] + f[2:,1:-1] + f[0:-2,1:-1] - 4*f[1:-1,1:-1])/dx**2
    return ret

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

def grad_dot_grad(a_mat, b_mat, h):
    ax,ay = grad(a_mat,h,h)
    bx,by = grad(b_mat,h,h)
    return ax*bx + ay*by

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

def ax_symmetry(u):
    x_sym, y_sym = False, False
    if(np.mean(np.abs(u - np.fliplr(u))) <= 1.0e-10 * np.mean(np.abs(u))):
        x_sym = True
    if(np.mean(np.abs(u - np.flipud(u))) <= 1.0e-10 * np.mean(np.abs(u))):
        y_sym = True
    return (x_sym, y_sym)

def anti_ax_symmetry(u):
    x_asym, y_asym = False, False
    if(np.mean(np.abs(u + np.fliplr(u))) <= 1.0e-10 * np.mean(np.abs(u))):
        x_asym = True
    if(np.mean(np.abs(u + np.flipud(u))) <= 1.0e-10 * np.mean(np.abs(u))):
        y_asym = True
    return (x_asym, y_asym)

def Hermitian(u):
    Hermitian = False
    if(np.mean(np.abs(u - np.transpose(u))) <= 1.0e-10 * np.mean(np.abs(u))):
        Hermitian = True
    return Hermitian

def get_frame(mesh_, lvl_func_):
    xmesh, ymesh = mesh_
    phi = lvl_func_(xmesh, ymesh)
    isOut = np.greater(phi,0)
    return 1-isOut

def print_error(u_result_, mesh_, sol_func_):
    xmesh,ymesh = mesh_
    dif = np.abs(u_result_ - sol_func_(xmesh,ymesh))
    L2Dif = L_n_norm(dif,2)
    maxDif = np.max(dif)
    print("Max error : ", maxDif)
    print("L^2 error : ", L2Dif)
    print("")
    return maxDif, L2Dif

def get_error(u_result_, mesh_, frame, sol_func_):
    xmesh,ymesh = mesh_
    dif = np.abs(u_result_ - sol_func_(xmesh,ymesh))
    L2Dif = L_n_norm_frame(dif,frame,2)
    maxDif = np.max(dif)
    print("Max error : ", maxDif)
    print("L^2 error : ", L2Dif)
    print("")
    return maxDif, L2Dif

def get_error_Neumann(u_result_, mesh_, frame, sol_func_):
    xmesh,ymesh = mesh_
    dif = np.abs(u_result_ - sol_func_(xmesh,ymesh))
    error = dif - np.mean(dif)
    dif = (u_result_ - sol_func_(xmesh,ymesh)) * frame
    error = (dif - np.sum(dif) / np.sum(frame)) * frame
    L2Dif = L_n_norm(error,2)
    maxDif = np.max(error)
    print("Max error (const added) : ", maxDif)
    print("L^2 error (const added) : ", L2Dif)
    print("")
    return maxDif, L2Dif

def plot3d_all(u_result_, mesh_, sol_func_,fig_label_, toPlot_ = [True, True, True, True]):
    xmesh, ymesh = mesh_
    sol_mesh = sol_func_(xmesh, ymesh)
    if(toPlot_[0]):
        #2D color plot of the max difference
#        fig_er = plt.figure("poisson solution error %d" % fig_label_)
#        plt.pcolor(xmesh, ymesh, (sol_mesh - u_result_)/(sol_mesh+1.0e-17))
        test_mat = (sol_mesh - u_result_)/(sol_mesh+1.0e-17)
        
        plt.matshow(test_mat )
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
