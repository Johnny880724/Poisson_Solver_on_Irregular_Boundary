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
    first_term = hf.laplace(J_mat,h,h) / (hf.abs_grad(phi_inv,h,h)**2 + singular_null)
    first_term -= (hf.laplace(K_mat,h,h) - J_mat*hf.laplace(phi_inv,h,h))*hf.laplace(phi_inv,h,h) / (hf.abs_grad(phi_inv,h,h)**4 + singular_null)
    second_term = np.heaviside(phi_inv,1)
    return Chi(phi_inv) * first_term + (1-Chi(phi_inv)) * second_term

def delta(phi_inv,h):
    I_mat = I(phi_inv)
    J_mat = J(phi_inv)
#    K_mat = K(phi)
    first_term = hf.laplace(I_mat,h,h) / (hf.abs_grad(phi_inv,h,h)**2 + singular_null)
    first_term -= (hf.laplace(J_mat,h,h) - I_mat*hf.laplace(phi_inv,h,h))*hf.laplace(phi_inv,h,h) / (hf.abs_grad(phi_inv,h,h)**4 + singular_null)
    return Chi(phi_inv) * first_term

def get_source(a, b, mesh, lvl_func_,f_mat_):
    xmesh, ymesh = mesh
    h = xmesh[0,1]-xmesh[0,0]
    phi = lvl_func_(xmesh,ymesh)
    #in the soruce term paper, the phi they use are inverted
    phi_inv = -phi
    
    #Discretization of the source term - formula (8)
    a_eff = a + (hf.norm_grad(a,(xmesh,ymesh),lvl_func_) - b) * phi_inv / (hf.abs_grad(phi_inv,h,h)+singular_null)
    S_mat = np.zeros_like(xmesh)
    H_mat = H(phi_inv,h)
    S_mat = hf.laplace(a_eff * H_mat,h,h) - H_mat * hf.laplace(a_eff,h,h) + H_mat * f_mat_
    
    #Discretization of the source term - formula (7)
#    S_mat = np.zeros_like(xmesh)
#    H_mat = H(phi_inv,h)
#    term1 = hf.laplace(a * H_mat,h,h)
#    term2 = - H_mat * hf.laplace(a, h, h)
#    term3 = - (b - hf.norm_grad(a,(xmesh,ymesh),lvl_func_)) * delta(phi_inv, h) * hf.abs_grad(phi_inv,h,h)
#    term4 = H_mat * f_mat_
#    S_mat = term1 + term2 + term3 + term4
    
    return S_mat

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
#    test_symmetry(tau_0,"tau_0")
#    test_symmetry(eps_0,"eps_0")
    tau = np.copy(tau_0)
    eps = np.copy(eps_0)
    tau_cur = np.copy(tau)
    eps_cur = np.copy(eps)
#    test_print = 0
#    print("before extpl: ", hf.ax_symmetry(val_))
    while(np.sum(tau) > 0):
#        print(hf.ax_symmetry(val_extpl))
#        plt.matshow(tau)
#        plt.title("tau")
#        plt.colorbar()
#        plt.matshow(eps)
#        plt.title("eps")
#        plt.colorbar()
#        test_print += 1
#        print("round %d" % test_print)
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
#        plt.matshow(val_extpl)
#        plt.title(str(test_print))
        
    return val_extpl

def setup_equations(bnd_type_, beta_p_val = 1.):
    
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
    
    global bnd_type
    bnd_type = bnd_type_
    
    ##below is for testing existing function purposes
    ######################################################################
    def desired_result(x, y):
        return (1 -0.25 * ((x - x0)**2 + (y - y0)**2))
    global desired_func 
    desired_func = desired_result
    
    def solution(x,y):
        sol = np.heaviside(-lvl_func(x,y),1)*(desired_func(x,y))
        return sol
    global sol_func
    sol_func = solution
    
    
    
    ##boundary conditions / jump conditions
    def jump_condition(x,y,u):
        u_desired = desired_func(x,y)
        u_n_desired = hf.norm_grad(u_desired,(x,y),lvl_func)
        #for Robin boundary condition only
        def sigma(x,y):
            return -0.25 * (hf.XYtoR(x-x0, y-y0)+10**-17)
        
        sigma_Robin = sigma(x, y)
        g_Robin = u_desired + sigma_Robin * u_n_desired
        #################################
        
        if(bnd_type == "Dirichlet"):
            a_mesh =  u_desired
            b_mesh =  beta_m * hf.grad_frame(u,(x,y),lvl_func)
            
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
    maxIterNum_ = 200000
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
        if(i % int(maxIterNum_/100) < 0.1):
            u_cur = u
            maxDif = np.max(np.abs(u_cur - u_prev)) / np.max(np.abs(u_cur))
            L2Dif = hf.L_n_norm(np.abs(u_cur - u_prev)) / hf.L_n_norm(u_cur)
            # check convergence and print process
            check_convergence_rate = 10**-11
            if(i % int(maxIterNum_/100) < 0.1):
                u_cur = u
                maxDif = np.max(np.abs(u_cur - u_prev)) / np.max(np.abs(u_cur))
                L2Dif = hf.L_n_norm(np.abs(u_cur - u_prev)) / hf.L_n_norm(u_cur)
                if(L2Dif < check_convergence_rate):
                    break;
                else:
                    u_prev = u_cur
                sys.stdout.write("\rProgress: %4g%%" % (i *100.0/maxIterNum_))
                sys.stdout.flush()
            sys.stdout.write("\rProgress: %4g%%" % (i *100.0/maxIterNum_))
            sys.stdout.flush()
        
    print("")
    return u

def test_symmetry(mat, mat_str):
    plt.matshow(mat)
    plt.title("this is " + mat_str)
    print( mat_str + " is symmetric : ",hf.ax_symmetry(mat))
    print( mat_str + " is anti-symmetric : ",hf.anti_ax_symmetry(mat))
    print( mat_str + " is Hermitian : ",hf.Hermitian(mat))

def poisson_jacobi_source_term_Dirichlet(u_init_, maxIterNum_, mesh_,lvl_func_,f_mesh_, g_mesh_):
    xmesh, ymesh = mesh_
    h = xmesh[0,1] - xmesh[0,0]
    phi = lvl_func_(xmesh, ymesh)
    grad_result = hf.grad(phi, h, h)
    nx = grad_result[0] / (np.sqrt(grad_result[0]**2 + grad_result[1]**2) + singular_null)
    ny = grad_result[1] / (np.sqrt(grad_result[0]**2 + grad_result[1]**2) + singular_null)
    
    #A4 compute u_x, u_y for Omega^- ^ N4 \ N1
    ux, uy = hf.grad(u_init_, h, h)
    
    #A5 Extrapolate u_x, u_y throughout N2
    N1 = get_N1(phi)
    N2 = get_N2(phi)
    Omega_m = np.heaviside(-phi, 1)
    eligible_0 = Omega_m * (1-N2)
    target_0 = N2 * (1-eligible_0)
    ux_extpl = extrapolation(ux, target_0, eligible_0)
    uy_extpl = extrapolation(uy, target_0, eligible_0)
#    test_symmetry(ux_extpl,"ux")
    
    #A6 compute the new a throughout N2
    a_mesh = - np.copy(g_mesh_)
    b_mesh = - (ux_extpl * nx + uy_extpl * ny)
#    test_symmetry(a_mesh, "a_mesh")
#    test_symmetry(b_mesh, "b_mesh")
#    plt.matshow(f_mesh_)
#    plt.colorbar()
    #A1 compute the source term
    source = get_source(-a_mesh, -b_mesh, (xmesh, ymesh), lvl_func_, f_mesh_)
#    test_symmetry(source,"source")
#    test_symmetry(source, "source")
    
    #A2 compute the source term with the addition of convergence term
    q = 0.75
    source += (q / h * u_init_) * (1-Omega_m) * N2
    
    #A3 call a Poisson solver resulting in u throughout Omega
    u_init = np.zeros_like(xmesh)
    u_result = poisson_jacobi_solver(u_init, 200000, (xmesh,ymesh), source)
    return u_result

if(__name__ == "__main__"):
    plt.close("all")
    
    setup_grid(65)
    setup_equations("Dirichlet")
    
    
    phi = lvl_func(xmesh, ymesh)
    phi_inv = -phi
    
    #1. Extend g(x,y) off of Gamma, define a, b throughout N2
#    xmesh_p = xmesh
#    ymesh_p = ymesh
    xmesh_p, ymesh_p = projection((xmesh,ymesh),phi_inv)
    g_ext = desired_func(xmesh_p, ymesh_p)
#    plt.matshow(g_ext)
#    print("is", hf.ax_symmetry(g_ext))
    
    #2. extrapolate f throughout N1 U Omega^+
    f_org = rhs_func(xmesh, ymesh)
    phi_inv = -phi
    N1 = get_N1(phi)
    Omega_m = np.heaviside(-phi, 1)
#    eligible_0 = Omega_m * (1-N1)
    eligible_0 = Omega_m
    target_0 = N1 * (1 - eligible_0)
    f_extpl = extrapolation(f_org, target_0, eligible_0)
#    plt.matshow(f_extpl)
#    print("is", hf.ax_symmetry(f_extpl))
    
    #3. initialize b = 0 throughout N2
    u_cur_result = np.copy(u_init)
    
    
    isOut = 1 - np.greater(phi,0)
    for i in range(10):
        u_result = poisson_jacobi_source_term_Dirichlet(u_cur_result, 200000, (xmesh,ymesh), lvl_func,f_extpl, g_ext)
        u_cur_result = np.copy(u_result)
        hf.print_error(u_result*isOut, (xmesh,ymesh),sol_func)
#        print(hf.ax_symmetry(u_result))
#        print(hf.Hermitian(u_result))
#    hf.print_error(u_result, (xmesh,ymesh),sol_func)
    plt.matshow(u_result)
    hf.plot3d_all(u_result, (xmesh,ymesh),sol_func,0)
    

def test(x,y):
    return x**2 + y**2
atest, btest = hf.grad(test(xmesh, ymesh), h, h)
