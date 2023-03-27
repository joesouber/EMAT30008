#%% Explicit euler method 
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix



def solve_pde(u_I, L, T, D, bound_left, bound_right ,boundary, method, mt=1000, mx=40):

    # Set up the numerical environment variables
    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    f_mesh = D*deltat/(deltax**2)    # mesh fourier number

    def matrix_dimensions(boundary):

        if boundary == 'dirichlet':
            dimension = mx - 1
            u_j = np.zeros(x.size)
            for i in range(0,len(u_j)):
                u_j[i] = u_I(x[i])
            
            return dimension, u_j


        if boundary == 'neumann':
            dimension = mx + 1
            u_j = np.zeros(x.size)
            for i in range(0,len(u_j)):
                u_j[i] = u_I(x[i])
            return dimension, u_j

    def A_dd_matrix(method, dimension, boundary):
        """ 
        In-bedded function which returns the appropriate tri-diagonal matrix in function 
        of the boudary condition and methid specified.
        """
        # Set tridiagonal matrix 
        # forward euler
        if method == 'forward':
            diag = [[f_mesh] * (dimension-1), [1 - 2*f_mesh] * dimension , [f_mesh] * (dimension-1)]
            A_dd = diags(diag, offsets = [-1,0,1], format = 'csc')
            if boundary == 'dirichlet':
                return A_dd, None

            elif boundary == 'neumann':
                A_dd = A_dd.toarray()
                A_dd[0,1] *= 2
                A_dd[-1,-2] *= 2   
                return csr_matrix(A_dd), None
            


