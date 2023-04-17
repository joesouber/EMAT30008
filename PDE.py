import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve
from scipy.optimize import fsolve

def tridiag_mat(pde_sol, BC_type, t, x, D,L,T=0.5,mt=1000,mx=100):
#possibly done
    delta_t = T / mt
    delta_x = L / mx

    n = len(pde_sol)

    subdiag = delta_t/(delta_x**2) * D*(x + delta_x/2)[1:]
    centre_diag = delta_t/(delta_x**2) * (D*(x+delta_x/2) + D*(x-delta_x/2))
    superdiag = delta_t/(delta_x**2) * D*(x-delta_x/2)[:-1]
    
    if BC_type == 'dirichlet':
        
        np.ones(n-1)[-1] = 0
        np.array([-1]*n)[0] = 0
        np.array([-1]*n)[-1] = 0

    elif BC_type == 'neumann':
        
        np.ones(n-1)[-1] = 2
    
    diagonals = (subdiag*np.ones(n-1), centre_diag*np.array([-1]*n), superdiag*np.ones(n-1))
    offset = (-1,0,1)
    
    tridiag_mat = diags(diagonals, offset, format='csr')
    return tridiag_mat