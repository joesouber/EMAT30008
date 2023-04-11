import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve
from scipy.optimize import fsolve

def construct_L(u, j, L, t, x, D, BC_type):
    delta_x = x[1] - x[0]
    delta_t = t[1] - t[0]

    subdiag = delta_t/(delta_x**2) * D(x + delta_x/2)[1:]
    diag = delta_t/(delta_x**2) * (D(x+delta_x/2) + D(x-delta_x/2))
    superdiag = delta_t/(delta_x**2) * D(x-delta_x/2)[:-1]

    off_diagonal_boundary_coefficient = np.ones(u.shape[0]-1)
    diag_boundary_coefficient = -np.ones(u.shape[0])

    boundary_coefficients = {
        'dirichlet': {
            'off_diagonal': 0,
            'diagonal': 0,
        },
        'neumann': {
            'off_diagonal': 2,
            'diagonal': -1,
        }
    }

    boundary_coef = boundary_coefficients[BC_type]
    off_diagonal_boundary_coefficient[-1] = boundary_coef['off_diagonal']
    diag_boundary_coefficient[0] = boundary_coef['diagonal']
    diag_boundary_coefficient[-1] = boundary_coef['diagonal']

    L = diags((subdiag*off_diagonal_boundary_coefficient, diag*diag_boundary_coefficient, superdiag*off_diagonal_boundary_coefficient), (-1,0,1), format='csr')

    return L