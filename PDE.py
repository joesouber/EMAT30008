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