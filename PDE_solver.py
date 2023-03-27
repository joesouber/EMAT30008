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

    def A_dd_matrix(method, matrix_dim, boundary):
        """ 
        In-bedded function which returns the appropriate tri-diagonal matrix in function 
        of the boudary condition and methid specified.
        """
        # Set tridiagonal matrix 
        # explicit euler
        if method == 'explicit':
            diag = [[f_mesh] * (matrix_dim-1), [1 - 2*f_mesh] * matrix_dim , [f_mesh] * (matrix_dim-1)]
            A_dd = diags(diag, offsets = [-1,0,1], format = 'csc')
            if boundary == 'dirichlet':
                return A_dd, None

            elif boundary == 'neumann':
                A_dd = A_dd.toarray()
                A_dd[0,1] *= 2
                A_dd[-1,-2] *= 2   
                return csr_matrix(A_dd), None
        
        

                   

    dimension, u_sol = matrix_dimensions(boundary)
    u_1 = np.zeros(mx+1)
    
   
    def b_dd(boundary):

        if boundary == 'dirichlet':
            return np.zeros(mx-1)
        if boundary == 'neumann':
            return np.zeros(mx+1)
        
    # setup additive vector and appropriate matrix dimensions
   
    dimension, u_sol = matrix_dimensions(boundary)
    u_1 = np.zeros(mx+1)
    B_dd = b_dd(boundary)
    diag1 = A_dd_matrix(method, dimension, boundary)
    
    # Solve PDE for each time value
    for i in range(0,mt):
        # forwad euler matrix calc
        if boundary == 'dirichlet':
            B_dd[0], B_dd[-1] = bound_left(t[i]), bound_right(t[i])
            if method == 'explicit':
                u_1[1:-1] = diag1.dot(u_sol[1:-1]) + B_dd * f_mesh


            # add boundary conditions
            u_1[0] = B_dd[0]
            u_1[-1] = B_dd[-1]
            #initialise u_j for the next time step
            u_sol[:] = u_1[:]


        if boundary == 'neumann':
            B_dd[0], B_dd[-1] = -bound_left(t[i]), bound_right(t[i])
            if method == 'explicit':
                u_1 = diag1.dot(u_sol) + 2 * f_mesh * deltax * B_dd
            





    return x, u_sol
            
            


