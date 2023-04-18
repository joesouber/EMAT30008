import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve
from scipy.optimize import fsolve

def apply_bc(pde_sol, time_step, Boundary_Cond, L, t, Boundary_type, CN = False,T=0.5,mt=1000,mx=100):
    delta_t = T / mt

    delta_x = L / mx

    #where alpha and beta represent the 0 and L bound conds.
    [alpha, beta] = [Boundary_Cond(i, t[time_step]) for i in [0, L]]

    if Boundary_type == 'dirichlet':
    # Add the boundary values to the solution vector
        pde_sol = np.append([alpha], pde_sol[1:-1])
        pde_sol = np.append(pde_sol, [beta])

    if Boundary_type == 'neumann':

        coefficients = [-2*delta_t/delta_x * alpha]  + [-2*delta_t/delta_x * beta] + [0]*(len(pde_sol)-2)
        pde_sol = [val + coef for val, coef in zip(pde_sol, coefficients)]

    return pde_sol

def tridiag_mat(pde_sol, Boundary_type, t, x, D,L,T=0.5,mt=1000,mx=100):
#possibly done
    delta_t = T / mt
    delta_x = L / mx

    n = len(pde_sol)

    subdiag = delta_t/(delta_x**2) * D*(x + delta_x/2)[1:]
    centre_diag = delta_t/(delta_x**2) * (D*(x+delta_x/2) + D*(x-delta_x/2))
    superdiag = delta_t/(delta_x**2) * D*(x-delta_x/2)[:-1]
    
    if Boundary_type == 'dirichlet':
        
        np.ones(n-1)[-1] = 0
        np.array([-1]*n)[0] = 0
        np.array([-1]*n)[-1] = 0

    elif Boundary_type == 'neumann':
        
        np.ones(n-1)[-1] = 2
    
    diagonals = (subdiag*np.ones(n-1), centre_diag*np.array([-1]*n), superdiag*np.ones(n-1))
    offset = (-1,0,1)
    
    tridiag_mat = diags(diagonals, offset, format='csr')
    return tridiag_mat


def explicit_euler(pde_sol, t, x, L, Boundary_Cond, Boundary_type, time_step, D, source_term=None, linearity='linear',T=0.5,mt=1000,mx=100):

    tridiag = tridiag_mat(pde_sol, Boundary_type, t, x, D,L,T=0.5,mt=1000,mx=100)
    mat1 = (identity(pde_sol.shape[0]) + tridiag)
    mat2= (pde_sol[:,time_step])
    b = (mat1).dot(mat2)
    delta_t = T / mt
    b += delta_t * source_term(x , t[time_step])
    pde_sol[:, time_step+1] = apply_bc(b, time_step, Boundary_Cond, L, t, Boundary_type)
    
    return pde_sol

def implicit_euler(pde_sol, t, x, L, Boundary_Cond, Boundary_type, time_step, D, source_term=None, linearity='linear',T=0.5,mt=1000,mx=100):


    delta_t = T / mt
    
    tridiag = tridiag_mat(pde_sol, Boundary_type, t, x, D,L,T=0.5,mt=1000,mx=100)
    n, _ = pde_sol.shape
    tridiag_identity_matrix = identity(n)-tridiag

    if linearity == 'linear':
        b = pde_sol.take(time_step, axis=1) + delta_t * source_term(x, t[time_step + 1])
        pde_sol[:, time_step + 1] = apply_bc(spsolve(tridiag_identity_matrix, b), time_step + 1, Boundary_Cond, L, t, Boundary_type)

    elif linearity == 'nonlinear':

        delta_t = T / mt
        delta_x = L / mx
        r = delta_t / (delta_x**2 * D)
    
        tridiag = tridiag_mat(pde_sol, Boundary_type, t, x, D,L,T=0.5,mt=1000,mx=100)
        n, _ = pde_sol.shape
        tridiag_identity_matrix = identity(n) - r * tridiag
    
        b = pde_sol[:, time_step] + delta_t * source_term(pde_sol[:, time_step], x, t[time_step + 1])
        pde_sol[:, time_step + 1] = apply_bc(spsolve(tridiag_identity_matrix, b), time_step + 1, Boundary_Cond, L, t, Boundary_type)

    return pde_sol

def crank_nicholson(pde_sol, t, x, L, Boundary_Cond, Boundary_type, time_step, D, source_term=None, linearity='linear',T=0.5,mt=1000,mx=100):

    delta_t = T / mt
    
    tridiag = tridiag_mat(pde_sol, Boundary_type, t, x, D,L,T=0.5,mt=1000,mx=100)
    n, _ = pde_sol.shape #trying ', _' notation.
    half_tridiag = 0.5*tridiag
    Identity_matrix = identity(n)
    
    cn_mat1 = (Identity_matrix + half_tridiag) 
    cn_mat2 = (Identity_matrix - half_tridiag) 

    b = cn_mat1.dot(pde_sol[:, time_step]) + (delta_t/2) * (source_term(x, t[time_step]) + source_term(x, t[time_step+1]))
    b = apply_bc(b, time_step, Boundary_Cond, L, t, Boundary_type, CN=True)



    pde_sol[:,time_step+1] = spsolve(cn_mat2, b)

    return pde_sol