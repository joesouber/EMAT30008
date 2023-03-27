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






