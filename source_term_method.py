# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 14:25:56 2019

@author: Johnny Tsao
"""
#import os,sys,inspect
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
#sys.path.insert(0,parentdir) 

import poisson_helper_functions as hf

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy import linalg as la
import matplotlib.animation as animation
import read_write_helper_functions as rw



#adding to denominator to avoid 0/0 error
singular_null = 1.0e-30

def setup_grid(N_grid_val = 100):
    # grid dimension
    grid_min = -1.
    grid_max = 1.
    
    global N_grid
    N_grid = N_grid_val
    # grid spacing
    global h
    h = (grid_max - grid_min) / (N_grid) 
    
    # define arrays to hold the x and y coordinates
    xy = np.linspace(grid_min,grid_max,N_grid+1)
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

def get_neighbor(domain):
    ret = np.zeros_like(domain)
    ret += domain
    ret[ 1:, :] += domain[:-1, :]
    ret[:-1, :] += domain[ 1:, :]
    ret[ :, 1:] += domain[ :,:-1]
    ret[ :,:-1] += domain[ :, 1:]
    
    return np.heaviside(ret,0)

def get_N1(phi):
    return Chi(phi)

def get_N2(phi):
    N1 = get_N1(phi)    
    return get_neighbor(N1)

def get_N3(phi):
    N2 = get_N2(phi)
    return get_neighbor(N2)

def H(phi_inv,h):
#    I_mat = I(phi)
    J_mat = J(phi_inv)
    K_mat = K(phi_inv)
    first_term_1 = hf.laplace(J_mat,h,h) / (hf.abs_grad(phi_inv,h,h)**2 + singular_null)
    first_term_2 = -(hf.laplace(K_mat,h,h) - J_mat*hf.laplace(phi_inv,h,h))*hf.laplace(phi_inv,h,h) / (hf.abs_grad(phi_inv,h,h)**4 + singular_null)
    first_term = first_term_1 + first_term_2
#    first_term -= hf.grad_dot_grad(J_mat, phi_inv,h)*hf.laplace(phi_inv,h,h) / (hf.abs_grad(phi_inv,h,h)**4 + singular_null)
    second_term = np.heaviside(phi_inv,1)
#    return np.heaviside(phi_inv,1)
    return Chi(phi_inv) * first_term + (1-Chi(phi_inv)) * second_term
#
def delta(phi_inv,h):
    I_mat = I(phi_inv)
    J_mat = J(phi_inv)
#    K_mat = K(phi)
    first_term = hf.laplace(I_mat,h,h) / (hf.abs_grad(phi_inv,h,h)**2 + singular_null)
    first_term -= (hf.laplace(J_mat,h,h) - I_mat*hf.laplace(phi_inv,h,h))*hf.laplace(phi_inv,h,h) / (hf.abs_grad(phi_inv,h,h)**4 + singular_null)
    return Chi(phi_inv) * first_term

def H_delta_test():
    setup_grid(101)
    setup_equations("exact")
    phi = lvl_func(xmesh, ymesh)
    plt.matshow(H(-phi,h))
    fig_dif = plt.figure()
    ax_dif = fig_dif.gca(projection='3d')
    ax_dif.plot_surface(xmesh, ymesh, H(-phi,h), cmap=cm.coolwarm)
#test2=H_delta_test()
    
def get_source(a, b, mesh, lvl_func_,f_mat_):
    xmesh, ymesh = mesh
    h = xmesh[0,1]-xmesh[0,0]
    phi = lvl_func_(xmesh,ymesh)
    #in the soruce term paper, the phi they use are inverted
    phi_inv = -phi
    
    #Discretization of the source term - formula (prev paper)
#    S_mat = np.zeros_like(xmesh)
#    H_h_mat = H(phi_inv,h)
#    H_mat = np.heaviside(phi_inv,1)
#    delta_mat = delta(phi_inv,h)
#    S_mat = a*hf.laplace(H_mat,h,h) - (hf.norm_grad(a,(xmesh,ymesh),lvl_func_) + b)*delta_mat*hf.abs_grad(phi_inv,h,h) + H_h_mat * f_mat_
    
    #Discretization of the source term - formula (8)
#    a_eff = (a + (hf.norm_grad(a,(xmesh,ymesh),lvl_func_) - b) * phi_inv / (hf.abs_grad(phi_inv,h,h)+singular_null))
#    H_h_mat = H(phi_inv,h)
#    H_mat = np.heaviside(phi_inv,1)
#    S_mat = hf.laplace(a_eff * H_mat,h,h) - H_h_mat * hf.laplace(a_eff,h,h) + H_h_mat * f_mat_
    
    #Discretization of the source term - formula (7)
    H_h_mat = H(phi_inv,h)
    H_mat = np.heaviside(phi_inv,1)
    term1 = hf.laplace(a * H_mat,h,h)
    term2 = - H_h_mat * hf.laplace(a, h, h)
    term3 = - (b - hf.norm_grad(a,(xmesh,ymesh),lvl_func_)) * delta(phi_inv, h) * hf.abs_grad(phi_inv,h,h)
    term4 = H_h_mat * f_mat_
    S_mat = term1 + term2 + term3 + term4
    
    return S_mat

def source_test():
    setup_grid(64)
    setup_equations("exact")
    phi = lvl_func(xmesh, ymesh)
    a_mesh = desired_func(xmesh,ymesh)
    b_mesh = hf.norm_grad(a_mesh, (xmesh,ymesh),lvl_func)
    source = get_source(a_mesh, b_mesh, (xmesh, ymesh), lvl_func, rhs_func(xmesh,ymesh))
    plt.pcolor(xmesh,ymesh,source)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    return H(-phi,h)

#source_test()
#projection algorithm
def projection(mesh_, phi_inv):
    xmesh, ymesh = mesh_
    h = xmesh[0,1]-xmesh[0,0]
    phi_abs_grad = hf.abs_grad(phi_inv,h,h)
    grad_tup = hf.grad(phi_inv,h,h)
    nx = -grad_tup[0] / (phi_abs_grad + singular_null)
    ny = -grad_tup[1] / (phi_abs_grad + singular_null)
    xp = xmesh + nx * phi_inv / (phi_abs_grad + singular_null)
    yp = ymesh + ny * phi_inv / (phi_abs_grad + singular_null)
    return xp, yp

#quadrature extrapolation algorithm
def extrapolation(val_, target_, eligible_):
    val_extpl = val_ * eligible_
    tau_0 = np.copy(target_)
    eps_0 = np.copy(eligible_)
    tau = np.copy(tau_0)
    eps = np.copy(eps_0)
    tau_cur = np.copy(tau)
    eps_cur = np.copy(eps)
#    print("before extpl: ", hf.ax_symmetry(val_))
    while(np.sum(tau) > 0):
        val_extpl_temp = np.copy(val_extpl)
        for i in range(len(val_)):
            for j in range(len(val_[i])):
                if(tau[i,j] == 1):
#                    print(i,j,"eps is ",eps[i,j])
                    triplet_count = 0
                    triplet_sum = 0
                    if(np.sum(eps[i+1:i+4,j]) > 2.9):
#                        print("right",eps[i+1:i+4,j])
                        triplet_count += 1
                        triplet_sum += 3*val_extpl[i+1,j] - 3*val_extpl[i+2,j] + val_extpl[i+3,j]
                    if(np.sum(eps[i-3:i,j]) > 2.9):
#                        print("left",eps[i-3:i,j])
                        triplet_count += 1
                        triplet_sum += 3*val_extpl[i-1,j] - 3*val_extpl[i-2,j] + val_extpl[i-3,j]
                    if(np.sum(eps[i,j+1:j+4]) > 2.9):
#                        print("up", eps[i,j+1:j+4])
                        triplet_count += 1
                        triplet_sum += 3*val_extpl[i,j+1] - 3*val_extpl[i,j+2] + val_extpl[i,j+3]
                    if(np.sum(eps[i,j-3:j]) > 2.9):
#                        print("down",i,j-3,j,eps[i,j-3:j])
                        triplet_count += 1
                        triplet_sum += 3*val_extpl[i,j-1] - 3*val_extpl[i,j-2] + val_extpl[i,j-3]
                    if(triplet_count > 0):
#                        print(" %d %d can leave" % (i,j))
                        val_extpl_temp[i,j] = triplet_sum / triplet_count
                        tau_cur[i,j] = 0
                        eps_cur[i,j] = 1
        tau = np.copy(tau_cur)
        eps = np.copy(eps_cur)
        val_extpl = np.copy(val_extpl_temp)
#        print("after extpl ", test_print, hf.ax_symmetry(val_extpl))
        
    return val_extpl

def setup_equations(beta_p_val = 1.):
    
    #offset center
    x0 = 0.0
    y0 = -0.0
    
    #Poisson equation coefficients
    global beta_p
    beta_p = beta_p_val
    global beta_m
    beta_m = 1.
    
    #rhs solution
    global r0
    r0 = 0.8
    def rhs(x,y):
#        return -np.zeros_like(x)
        return -np.ones_like(x)
    
    global rhs_func
    rhs_func = rhs
    
    
    def level(x,y):
        return np.array(hf.XYtoR(x,y) - r0)
    
    global lvl_func
    lvl_func = level
    
    
    ##below is for testing existing function purposes
    ######################################################################
    def desired_result(x, y):
        return (1 - 0.25 * ((x - x0)**2 + (y - y0)**2))
    global desired_func 
    desired_func = desired_result
    
    def solution(x,y):
        sol = np.heaviside(-lvl_func(x,y),1)*(desired_func(x,y))
        return sol
    global sol_func
    sol_func = solution
    
    def norm_der(x,y):
        u_x = - 0.5 * (x - x0)
        u_y = - 0.5 * (y - y0)
        n_x = x / (hf.XYtoR(x, y)+singular_null)
        n_y = y / (hf.XYtoR(x, y)+singular_null)
        u_n = (u_x * n_x + u_y * n_y) / (hf.XYtoR(n_x,n_y)+singular_null)
        return u_n
    global desired_func_n
    desired_func_n = norm_der
    
    #for Robin boundary condition
    def sigma(x,y):
        theta = hf.XYtoTheta(x,y)
        return 2.5 + 0.25 * x * np.sin(3*theta)
    global sigma_func
    sigma_func = sigma
    
#No 1, 5 in the source term paper
def setup_equations_2(beta_p_val = 1.):
    A = 0.5 * np.pi
    B = 0.75 * np.pi
    alpha = 4
    beta = 1.25
    def rhs(x,y):
#        return -np.zeros_like(x)
        return -(A**2 + B**2) * np.sin(A*x) * np.cos(B*y)
    
    global rhs_func
    rhs_func = rhs
    
    
    def level(x,y):
        return -0.9 + hf.XYtoR(alpha*x, beta*y)
    
    global lvl_func
    lvl_func = level
    
    
    ##below is for testing existing function purposes
    ######################################################################
    def desired_result(x, y):
        return 1 + np.sin(A*x) * np.cos(B*y)
    global desired_func 
    desired_func = desired_result
    
    def solution(x,y):
        sol = np.heaviside(-lvl_func(x,y),1)*(desired_func(x,y))
        return sol
    global sol_func
    sol_func = solution
    
    def norm_der(x,y):
        u_x = A * np.cos(A*x) * np.cos(B*y)
        u_y = - B * np.sin(A*x) * np.sin(B*y)
        n_x = alpha**2 * x / (hf.XYtoR(alpha*x, beta*y)+singular_null)
        n_y = beta**2 * y / (hf.XYtoR(alpha*x, beta*y)+singular_null)
        u_n = (u_x * n_x + u_y * n_y) / (hf.XYtoR(n_x,n_y)+singular_null)
        return u_n
    global desired_func_n
    desired_func_n = norm_der
    
    #for Robin boundary condition
    def sigma(x,y):
        theta = hf.XYtoTheta(x,y)
        return 2.5 + 0.25 * x * np.sin(3*theta)
    
    global sigma_func
    sigma_func = sigma
    
#No 2, 6, 9 in the source term paper
def setup_equations_3(beta_p_val = 1.):
    A = 2.7
    B = 3.1
    
    #level function
    coef = 0.1
    const = 0.5
    theta_coef = 4.
    theta_const = 0.17 * np.pi
    beta = 1.
    def rhs(x,y):
#        return -np.zeros_like(x)
        return -(A**2 + B**2) * np.sin(A*x) * np.cos(B*y)
    
    global rhs_func
    rhs_func = rhs
    
    
    def level(x,y):
        theta = np.arctan(y/(x+singular_null))
        r = hf.XYtoR(x, y)
        return -(const + coef * np.sin(theta_coef*theta + theta_const) - r)
    
    global lvl_func
    lvl_func = level
    
    
    ##below is for testing existing function purposes
    ######################################################################
    def desired_result(x, y):
        return 1 + np.sin(A*x) * np.cos(B*y)
    global desired_func 
    desired_func = desired_result
    
    def solution(x,y):
        sol = np.heaviside(-lvl_func(x,y),1)*(desired_func(x,y))
        return sol
    global sol_func
    sol_func = solution
    
    def norm_der(x,y):
        theta = hf.XYtoTheta(x, y)
        r = hf.XYtoR(x, y)
        u_x = A * np.cos(A*x) * np.cos(B*y)
        u_y = - B * np.sin(A*x) * np.sin(B*y)
        dphi_dr = 1.
        dphi_dtheta = -coef * theta_coef * np.cos(theta_coef*theta + theta_const)
        n_x = np.cos(theta) * dphi_dr - np.sin(theta) * dphi_dtheta / (r + singular_null)
        n_y = np.sin(theta) * dphi_dr + np.cos(theta) * dphi_dtheta / (r + singular_null)
        u_n = (u_x * n_x + u_y * n_y) / (hf.XYtoR(n_x,n_y)+singular_null)
        return u_n
    global desired_func_n
    desired_func_n = norm_der
    
    #for Robin boundary condition
    def sigma(x,y):
        theta = hf.XYtoTheta(x,y)
        return 2.5 + 0.25 * x * np.sin(3*theta)
    global sigma_func
    sigma_func = sigma
    
def test_projection():
#    phi = hf.XYtoR(xmesh, ymesh) - r0
    phi = np.exp(-xmesh**2 - ymesh**2) - np.exp(-r0**2)
    plt.matshow(phi)
    phi_inv = -phi
    xp, yp = projection((xmesh, ymesh), phi)
    N1 = Chi(phi)
    plt.matshow(N1*desired_func(xmesh, ymesh))
    plt.colorbar()
    plt.matshow(N1*desired_func(xp, yp))
    plt.colorbar()
    plt.matshow(N1*(desired_func(xp, yp)-desired_func(xmesh, ymesh)))
    plt.colorbar()
    print(hf.ax_symmetry(N1*(desired_func(xp, yp)-desired_func(xmesh, ymesh))))
#test_projection()

def test_extrapolation():
    setup_grid()
    setup_equations("exact")
    phi = lvl_func(xmesh, ymesh)
    omega_m = np.heaviside(-phi,0)
    N1 = get_N1(phi)
    N2 = get_N2(phi)
    eps_0 = omega_m * (1-N1)
    tau_0 = N1
    f_org = np.exp(-xmesh**2 - ymesh**2)
    f_final = extrapolation(f_org, tau_0, eps_0)
    plt.matshow(f_org)
    plt.colorbar()
    plt.matshow(f_final)
    plt.colorbar()
    plt.matshow(N2)
    plt.colorbar()
    
    
#test_extrapolation()

def poisson_jacobi_solver(u_init_,maxIterNum_, mesh_, source_):
    u_prev = u_init_
    u = u_init_
    global iterNum_record
    iterNum_record = 0
    for i in range(maxIterNum_):
        iterNum_record += 1
        # enforce boundary condition
        u[ 0, :] = np.zeros_like(u[ 0, :])
        u[-1, :] = np.zeros_like(u[-1, :])
        u[ :, 0] = np.zeros_like(u[ :, 0])
        u[ :,-1] = np.zeros_like(u[ :,-1])
    
        u_new = np.copy(u)
    
        # update u according to Jacobi method formula
        # https://en.wikipedia.org/wiki/Jacobi_method
        
        del_u = u[1:-1,2:] + u[1:-1,0:-2] + u[2:,1:-1] + u[0:-2,1:-1]
        u_new[1:-1,1:-1] = -h**2/4 * (source_[1:-1,1:-1] - del_u/h**2)
        u = u_new
        # check convergence and print process
        check_convergence_rate = 10**-11
        if(i % int(maxIterNum_*0.01) < 0.1):
            u_cur = u
            maxDif = np.max(np.abs(u_cur - u_prev)) / np.max(np.abs(u_cur))
            L2Dif = hf.L_n_norm(np.abs(u_cur - u_prev)) / hf.L_n_norm(u_cur)
            if(L2Dif < check_convergence_rate):
                break;
            else:
                u_prev = u_cur
#            sys.stdout.write("\rProgress: %4g%%" % (i *100.0/maxIterNum_))
            sys.stdout.write("\rProgress: %4d out of %4d" % (i,maxIterNum_))
            sys.stdout.flush()
        
    print("")
    return u

def test_symmetry(mat, mat_str):
    plt.matshow(mat)
    plt.title("this is " + mat_str)
    print( mat_str + " is symmetric : ",hf.ax_symmetry(mat))
    print( mat_str + " is anti-symmetric : ",hf.anti_ax_symmetry(mat))
    print( mat_str + " is Hermitian : ",hf.Hermitian(mat))

def poisson_jacobi_source_term_Dirichlet(u_init_, maxMultiple_, mesh_,lvl_func_,rhs_func_,\
                                         desired_func_, iteration_total, switches = [False,False]):
    xmesh, ymesh = mesh_
    h = xmesh[0,1] - xmesh[0,0]
    N = len(xmesh)
    phi = lvl_func_(xmesh, ymesh)
    phi_inv = -phi
    grad_result = hf.grad(phi, h, h)
    nx = grad_result[0] / (np.sqrt(grad_result[0]**2 + grad_result[1]**2) + singular_null)
    ny = grad_result[1] / (np.sqrt(grad_result[0]**2 + grad_result[1]**2) + singular_null)
    maxIterNum_ = maxMultiple_ * N**2
    
    N1 = get_N1(phi)
    N2 = get_N2(phi)
    Omega_m = np.heaviside(-phi, 1)
    isOut = np.greater(phi,0)
    
    #optional switch for animation, data-writing
    animation_switch = switches[0]
    write_data_switch = switches[1]
    
    #1. Extend g(x,y) off of Gamma, define a throughout N2
    xmesh_p, ymesh_p = projection((xmesh,ymesh),phi_inv)
    g_ext = desired_func_(xmesh_p, ymesh_p)
    a_mesh = -g_ext * N2
    
    #2. extrapolate f throughout N1 U Omega^+
    f_org = rhs_func_(xmesh, ymesh)
    eligible_0 = Omega_m * (1-N1)
    target_0 = N1 * (1 - eligible_0)
    f_extpl = extrapolation(f_org, target_0, eligible_0)
    
    #3. initialize b = 0 throughout N2
    b_mesh = np.zeros_like(u_init_)
    u_cur_result = np.copy(u_init_)
    
    def sol_func(x,y):
        return desired_func_(x,y) * (1-isOut)
    sol = sol_func
    
    if(animation_switch):
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        ax.set_zlim3d(0, 1.0)
        plots = []
        
    if(write_data_switch):
        maxDif_array = []
        it_array = []
        L2Dif_array = []
    
    #termination array
    Q_array = np.zeros(iteration_total)
    
    for it in range(iteration_total):
        print("This is iteration %d :" % (it + 1))
        
        #A1 compute the source term
        source = get_source(-a_mesh, -b_mesh, (xmesh, ymesh), lvl_func_, f_extpl)
        
        #A2 compute the source term with the addition of convergence term
        q = min(it,0.75)
        source += (q / h * u_cur_result) * (1-Omega_m) * N2
            
        #A3 call a Poisson solver resulting in u throughout Omega
        u_result = poisson_jacobi_solver(u_init_, maxIterNum_, (xmesh,ymesh), source)
        maxDif,L2Dif = hf.get_error(u_result, (xmesh, ymesh), 1-isOut, sol)
        u_cur_result = np.copy(u_result)
        
        #A4 compute u_x, u_y for Omega^- ^ N4 \ N1
        ux, uy = hf.grad(u_result, h, h)
        
        #A5 Extrapolate u_x, u_y throughout N2
        eligible_0 = Omega_m * (1-N1)
        target_0 = N1 * (1-eligible_0)
        ux_extpl = extrapolation(ux, target_0, eligible_0)
        uy_extpl = extrapolation(uy, target_0, eligible_0)
        
        #A6 compute the new a throughout N2
        b_mesh = - (ux_extpl * nx + uy_extpl * ny)
        
        #A7 check for termination
        Q_array[it] = np.max(u_result * isOut * N2)
        if(it > 5):
            if(Q_array[it] >= 0.99 * Q_array[it-1]):
                break
        
        if(animation_switch):
            plot = ax.plot_surface(xmesh,ymesh,u_result, animated=True, cmap=cm.coolwarm)
            plots.append([plot])
            
        if(write_data_switch):
            it_array.append(it)
            maxDif_array.append(maxDif)
            L2Dif_array.append(L2Dif)
        
        #for poster
#        fig_poster = plt.figure()
#        plt.rcParams.update({'font.size': 14})
#        ax_poster = fig_poster.gca(projection = '3d')
#        ax_poster.plot_surface(xmesh,ymesh,u_result, cmap=cm.coolwarm)
#        ax_poster.set_title("Iteration = %d" % (it+1))
#        ax_poster.set_xlabel("x")
#        ax_poster.set_ylabel("y")
#        ax_poster.set_zlabel("u")
#        ax_poster.set_zlim3d(0,2.0)
        
    if(write_data_switch):
        #write data of the error
        rw.write_float_data("\\source_term_method\\Dirichlet\\Dirichlet_"+str(N-1),[it_array,maxDif_array,L2Dif_array])
    if(animation_switch):
        #animate the plots
        ani = animation.ArtistAnimation(fig, plots, interval=300, blit=True,repeat_delay=0)
        return ani
        
    return u_result

def poisson_jacobi_source_term_Neumann(u_init_, maxMultiple_, mesh_,lvl_func_,rhs_func_,\
                                       desired_func_, desired_func_n_, iteration_total,switches=[False,False]):
    xmesh, ymesh = mesh_
    h = xmesh[0,1] - xmesh[0,0]
    N = len(xmesh)
    phi = lvl_func_(xmesh, ymesh)
    phi_inv = -phi
    
    N1 = get_N1(phi)
    N2 = get_N2(phi)
    Omega_m = np.heaviside(-phi, 1)
    isOut = np.greater(phi,0)
    maxIterNum = maxMultiple_ * N**2
    
    #optional switch for animation, data-writing
    animation_switch = switches[0]
    write_data_switch = switches[1]
    
    #1. Extend g(x,y) off of Gamma, define b throughout N2
    xmesh_p, ymesh_p = projection((xmesh,ymesh),phi_inv)
    g_ext = desired_func_n_(xmesh_p, ymesh_p)
    b_mesh = -g_ext * N2
    
    #2. extrapolate f throughout N1 U Omega^+
    f_org = rhs_func_(xmesh, ymesh)
    eligible_0 = Omega_m * (1-N1)
    target_0 = N1 * (1 - eligible_0)
    f_extpl = extrapolation(f_org, target_0, eligible_0)
    
    #3. initialize a = 0 throughout N2
    a_mesh = - u_init_
    u_cur_result = np.copy(u_init_)
    
    def sol_func(x,y):
        return desired_func_(x,y) * (1-isOut)
    sol = sol_func
    
    if(animation_switch):
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        ax.set_zlim3d(0, 1.0)
        plots = []
        
    if(write_data_switch):
        maxDif_array = []
        it_array = []
        L2Dif_array = []
    
    #termination array
    Q_array = np.zeros(iteration_total)
    
    for it in range(iteration_total):
        print("This is iteration %d :" % (it + 1))
        #A1-1 compute the source term
        source = get_source(-a_mesh, -b_mesh, (xmesh, ymesh), lvl_func_, f_extpl)
        
        #A1-2 compute the source term with the addition of convergence term
        q = -0.75 * min(1,it*0.1)
        source += (q / h * u_cur_result) * (1-Omega_m) * N2
            
        #A2 call a Poisson solver resulting in u throughout Omega
        u_result = poisson_jacobi_solver(u_init_, maxIterNum, (xmesh,ymesh), source)
        maxDif,L2Dif = hf.get_error_Neumann(u_result, (xmesh, ymesh), 1-isOut, sol)
        u_cur_result = np.copy(u_result)
    
        #A3-1 Extrapolate u throughout N2
        eligible_0 = Omega_m * (1-N1)
        target_0 = N2 * (1-eligible_0)
        u_extpl = extrapolation(u_result, target_0, eligible_0)
        
        #A3-2 compute the new a throughout N2
        a_mesh = - np.copy(u_extpl)
        
        #A4 check for termination
        Q_array[it] = np.max(u_result * isOut * N2)
        if(it > 5):
            if(Q_array[it] >= 0.99 * Q_array[it-1]):
                break
        
        if(animation_switch):
            plot = ax.plot_surface(xmesh,ymesh,u_result, animated=True, cmap=cm.coolwarm)
            plots.append([plot])
            
        if(write_data_switch):
            it_array.append(it)
            maxDif_array.append(maxDif)
            L2Dif_array.append(L2Dif)
            
    if(write_data_switch):
        #write data of the error
        rw.write_float_data("\\source_term_method\\Neumann\\Neumann_"+str(N-1),[it_array,maxDif_array,L2Dif_array])
    if(animation_switch):
        #animate the plots
        ani = animation.ArtistAnimation(fig, plots, interval=300, blit=True,repeat_delay=0)
        return ani
        
    return u_result

def poisson_jacobi_source_term_Robin(u_init_, maxMultiple_, mesh_,lvl_func_,rhs_func_,desired_func_\
                                     ,desired_func_n_,sigma_func_,iteration_total,switches=[False,False]):
    xmesh, ymesh = mesh_
    h = xmesh[0,1] - xmesh[0,0]
    N = len(xmesh)
    maxIterNum = maxMultiple_ * N**2
    phi = lvl_func_(xmesh, ymesh)
    phi_inv = -phi
    
    N1 = get_N1(phi)
    N2 = get_N2(phi)
    Omega_m = np.heaviside(-phi, 1)
    isOut = np.greater(phi,0)
    
    #optional switch for animation, data-writing
    animation_switch = switches[0]
    write_data_switch = switches[1]
    
    #1. Extend sigma(x,y) off of Gamma, define sigma throughout N2
    xmesh_p, ymesh_p = projection((xmesh,ymesh),phi_inv)
    sigma_mesh = sigma_func_(xmesh_p, ymesh_p)
    
    #2. Extend g(x,y) off of Gamma, define b throughout N2
    g_ext = desired_func_n_(xmesh_p, ymesh_p) + sigma_mesh * desired_func_(xmesh_p, ymesh_p)
    plt.matshow(g_ext*N2)
    g_mesh = g_ext * N2
    
    #3. extrapolate f throughout N1 U Omega^+
    f_org = rhs_func_(xmesh, ymesh)
    eligible_0 = Omega_m * (1-N1)
    target_0 = N1 * (1 - eligible_0)
    f_extpl = extrapolation(f_org, target_0, eligible_0)
    
    #4. initialize a = 0 throughout N2
    a_mesh = - u_init_
    u_cur_result = np.copy(u_init_)
    
    def sol_func(x,y):
        return desired_func_(x,y) * (1-isOut)
    sol = sol_func
    
    if(animation_switch):
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        ax.set_zlim3d(0, 1.0)
        plots = []
        
    if(write_data_switch):
        maxDif_array = []
        it_array = []
        L2Dif_array = []
    
    #termination array
    Q_array = np.zeros(iteration_total)
    
    
    for it in range(iteration_total):
        print("This is iteration %d :" % (it + 1))
        #A1 compute b = - (g + sigma * a)
        ##  sigma * u + u_n = g
        b_mesh = - (g_mesh + a_mesh * (sigma_mesh + singular_null)) 
        
        #A2 compute the source term
        source = get_source(-a_mesh, -b_mesh, (xmesh, ymesh), lvl_func_, f_extpl)
        
        #A2-2 compute the source term with the addition of convergence term
        q = -0.75 * min(1,it*0.1)
        source += (q / h * u_cur_result) * (1-Omega_m) * N2
            
        #A3 call a Poisson solver resulting in u throughout Omega
        u_result = poisson_jacobi_solver(u_init_, maxIterNum, (xmesh,ymesh), source)
        maxDif,L2Dif = hf.get_error(u_result, (xmesh, ymesh), 1-isOut, sol)
        u_cur_result = np.copy(u_result)
    
        #A4-1 Extrapolate u throughout N2
        eligible_0 = Omega_m * (1-N1)
        target_0 = N2 * (1-eligible_0)
        u_extpl = extrapolation(u_result, target_0, eligible_0)
        
        #A4-2 compute the new a throughout N2
        a_mesh = - np.copy(u_extpl)
        
        #A5 check for termination
        Q_array[it] = np.max(u_result * isOut * N2)
        if(it > 5):
            if(Q_array[it] >= 0.99 * Q_array[it-1]):
                break
        
        #for poster
#        fig_poster = plt.figure()
#        plt.rcParams.update({'font.size': 14})
#        ax_poster = fig_poster.gca(projection = '3d')
#        ax_poster.plot_surface(xmesh,ymesh,u_result, cmap=cm.coolwarm)
#        ax_poster.set_title("Iteration = %d" % (it+1))
#        ax_poster.set_xlabel("x")
#        ax_poster.set_ylabel("y")
#        ax_poster.set_zlabel("u")
#        ax_poster.set_zlim3d(0,2.0)
        
        if(animation_switch):
            plot = ax.plot_surface(xmesh,ymesh,u_result, animated=True, cmap=cm.coolwarm)
            plots.append([plot])
            
        if(write_data_switch):
            it_array.append(it)
            maxDif_array.append(maxDif)
            L2Dif_array.append(L2Dif)
        
    if(write_data_switch):
        #write data of the error
        rw.write_float_data("\\source_term_method\\Robin\\Robin_"+str(N-1),[it_array,maxDif_array,L2Dif_array])
    if(animation_switch):
        #animate the plots
        ani = animation.ArtistAnimation(fig, plots, interval=300, blit=True,repeat_delay=0)
        return ani
    
    return u_result

if(__name__ == "__main__"):
    plt.close("all")
    
    setup_grid(64)
    setup_equations_3()
    
    ##show analytical plot
#    fig_an = plt.figure()
#    ax_an = fig_an.gca(projection = '3d')
#    plot = ax_an.plot_surface(xmesh,ymesh,sol_func(xmesh,ymesh), cmap=cm.coolwarm)
    
#    u_result = poisson_jacobi_source_term_Dirichlet(u_init, 10, (xmesh,ymesh), lvl_func,rhs_func,desired_func,30,[True,False])
#    u_result = poisson_jacobi_source_term_Neumann(u_init, 10, (xmesh,ymesh), lvl_func,rhs_func,desired_func,desired_func_n,30,[False,True])
#    u_result = poisson_jacobi_source_term_Robin(u_init, 10, (xmesh,ymesh), lvl_func,rhs_func,desired_func,desired_func_n,sigma_func,30,[True,True])
   


    #convergence plot
#    iter_num_array = np.array([2**(i+5) for i in range(6)],dtype = int)
#    iter_num_array = np.array([32,50],dtype = int)
#    maxDif_array = np.zeros_like(iter_num_array)
#    for it_conv in range(len(iter_num_array)):
#        cur_grid_size = iter_num_array[it_conv]
#        setup_grid(cur_grid_size)
#        setup_equations_2()
#        mesh = (xmesh, ymesh)
#        u_result,ani = poisson_jacobi_source_term_Dirichlet(u_init, 10, mesh, lvl_func,rhs_func,desired_func,50)
#        print("Error for : %d * %d grid" % (cur_grid_size,cur_grid_size))
#        maxDif, L2Dif = hf.get_error(u_result, mesh, hf.get_frame(mesh, lvl_func), sol_func)
#        
#    plt.plot(np.log(iter_num_array), np.log(np.log(maxDif_array)))
        
        
        