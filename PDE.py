import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve
from scipy.optimize import fsolve

def apply_bc(pde_sol, time_step, Boundary_Cond, L, t, Boundary_type, CN=False, T=0.5, mt=1000, mx=100):
    """
    Applies boundary conditions to the solution vector of a PDE.

    Args:
        pde_sol (ndarray): The solution vector of the PDE.
        time_step (int): The current time step.
        Boundary_Cond (function): The boundary condition function.
        L (float): The length of the domain.
        t (ndarray): The time vector.
        Boundary_type (str): The type of boundary condition.
        CN (bool): Whether the Crank-Nicolson method is being used. Defaults to False.
        T (float): The final time. Defaults to 0.5.
        mt (int): The number of time steps. Defaults to 1000.
        mx (int): The number of spatial steps. Defaults to 100.

    Returns:
        ndarray: The solution vector with the boundary conditions applied.
    """

    delta_t = T / mt
    delta_x = L / mx

    #where alpha and beta represent the 0 and L bound conds.
    [alpha, beta] = [Boundary_Cond(i, t[time_step]) for i in [0, L]]

    if Boundary_type == 'dirichlet':
        # Add the boundary values to the solution vector
        pde_sol = np.append([alpha], pde_sol[1:-1]) 
        pde_sol = np.append(pde_sol, [beta]) 

    if Boundary_type == 'neumann':
        coefficients = [-2*delta_t/delta_x * alpha]  + [-2*delta_t/delta_x * beta] + [0]*(len(pde_sol)-2) #this is the same as the dirichlet case but with the boundary conditions added to the rhs
        pde_sol = [val + coef for val, coef in zip(pde_sol, coefficients)]  #expermimenting with a list comprehension
    
    elif Boundary_type == 'robin':
        # Calculate the Robin coefficient at the boundary
        r0 = Boundary_Cond(0, t[time_step], robin=True)
        rL = Boundary_Cond(L, t[time_step], robin=True)

        # Modify the matrix coefficients to account for Robin boundary conditions
        if CN:
            pde_sol[0] += (r0*delta_t/delta_x)*pde_sol[1]
            pde_sol[-1] += (rL*delta_t/delta_x)*pde_sol[-2]
        else:
            pde_sol[0] += (2*r0*delta_t/delta_x)*pde_sol[0] + (2*delta_t/delta_x)*alpha
            pde_sol[-1] += (2*rL*delta_t/delta_x)*pde_sol[-1] + (2*delta_t/delta_x)*beta

    return pde_sol


def tridiag_mat(pde_sol, Boundary_type, t, x, D, L, linearity, T=0.5, mt=1000, mx=100):
    """
    Constructs the tridiagonal matrix for a given PDE.

    Args:
            pde_sol (ndarray): The solution vector of the PDE.
            Boundary_type (str): The type of boundary condition.
            t (ndarray): The time vector.
            x (ndarray): The spatial vector.
            D (function or ndarray): The diffusion coefficient. Can be a function or an ndarray.
            L (float): The length of the domain.
            linearity (str): The linearity of the PDE. Can be 'linear' or 'nonlinear'.
            T (float): The final time. Defaults to 0.5.
            mt (int): The number of time steps. Defaults to 1000.
            mx (int): The number of spatial steps. Defaults to 100.

    Returns:
        csr_matrix: The tridiagonal matrix.
    """

    delta_t = T / mt
    delta_x = L / mx
    DXDT2 = delta_t/(delta_x**2)
    n = len(pde_sol)
    
    # Setting up the coeffecients of the diagonals, so that the matrix 
    # can be constructed for the variable diffusion coefficient case.

    if linearity == 'linear':
        subdiag = DXDT2 * D(x + delta_x/2)[1:]
        centre_diag = DXDT2 * (D(x+delta_x/2) + D(x-delta_x/2))
        superdiag = DXDT2 * D(x-delta_x/2)[:-1]

    elif linearity == 'nonlinear':
        subdiag = DXDT2 * D*(x + delta_x/2)[1:]
        centre_diag = DXDT2 * (D*(x+delta_x/2) + D*(x-delta_x/2))
        superdiag = DXDT2 * D*(x-delta_x/2)[:-1]

    if Boundary_type == 'dirichlet':
        # Modify the first and last rows of the matrix for Dirichlet boundary conditions
        np.ones(n-1)[-1] = 0
        np.array([-1]*n)[0] = 0
        np.array([-1]*n)[-1] = 0

    elif Boundary_type == 'neumann':
        # Modify the last row of the matrix for Neumann boundary conditions
        np.ones(n-1)[-1] = 2

    elif Boundary_type == 'robin':
        # Modify the matrix coefficients for Robin boundary conditions
        beta = 0
        alpha = 0
        gamma = 0
        delta = 0

        subdiag[0] += DXDT2 * alpha / delta_x
        centre_diag[0] += DXDT2 * (-2 * alpha / delta_x + beta)
        superdiag[0] += DXDT2 * alpha / delta_x
        pde_sol[0] += DXDT2 * gamma

        subdiag[-1] += DXDT2 * beta / delta_x
        centre_diag[-1] += DXDT2 * (2 * beta / delta_x + alpha)
        superdiag[-1] += DXDT2 * beta / delta_x
        pde_sol[-1] += DXDT2 * delta

    diagonals = (subdiag * np.ones(n-1), centre_diag * np.array([-1] * n), superdiag * np.ones(n-1))
    offset = (-1, 0, 1)

    tridiag_mat = diags(diagonals, offset, format='csr')

    return tridiag_mat


def explicit_euler(pde_sol, t, x, L, Boundary_Cond, Boundary_type, time_step, D, linearity, source_term=None, T=0.5, mt=1000, mx=100):
    """
    Solves a PDE using the explicit Euler method.

    Args:
        pde_sol (ndarray): The solution vector of the PDE.
        t (ndarray): The time vector.
        x (ndarray): The spatial vector.
        L (float): The length of the domain.
        Boundary_Cond (function): The boundary condition function.
        Boundary_type (str): The type of boundary condition.
        time_step (int): The current time step.
        D (function or ndarray): The diffusion coefficient. Can be a function or an ndarray.
        linearity (str): The linearity of the PDE. Can be 'linear' or 'nonlinear'.
        source_term (function or ndarray): The source term. Can be a function or an ndarray. Defaults to None.
        T (float): The final time. Defaults to 0.5.
        mt (int): The number of time steps. Defaults to 1000.
        mx (int): The number of spatial steps. Defaults to 100.

    Returns:
        ndarray: The updated solution vector.
    """

    # Construct the tridiagonal matrix
    tridiag = tridiag_mat(pde_sol, Boundary_type, t, x, D, L, linearity, T=0.5, mt=1000, mx=100)
    mat1 = (identity(pde_sol.shape[0]) + tridiag)
    mat2 = (pde_sol[:, time_step])

    # Compute the right-hand side of the equation
    b = (mat1).dot(mat2)
    delta_t = T / mt
    if source_term is not None:
        b += delta_t * source_term(x, t[time_step])

    # Apply boundary conditions to the solution vector
    pde_sol[:, time_step+1] = apply_bc(b, time_step, Boundary_Cond, L, t, Boundary_type)

    return pde_sol


def implicit_euler(pde_sol, t, x, L, Boundary_Cond, Boundary_type, time_step, D, linearity, source_term=None, T=0.5, mt=1000, mx=100):
    """
    Solves a PDE using the implicit Euler method.

    Args:
        pde_sol (ndarray): The solution vector of the PDE.
        t (ndarray): The time vector.
        x (ndarray): The spatial vector.
        L (float): The length of the domain.
        Boundary_Cond (function): The boundary condition function.
        Boundary_type (str): The type of boundary condition.
        time_step (int): The current time step.
        D (function or ndarray): The diffusion coefficient. Can be a function or an ndarray.
        linearity (str): The linearity of the PDE. Can be 'linear' or 'nonlinear'.
        source_term (function or ndarray): The source term. Can be a function or an ndarray. Defaults to None.
        T (float): The final time. Defaults to 0.5.
        mt (int): The number of time steps. Defaults to 1000.
        mx (int): The number of spatial steps. Defaults to 100.

    Returns:
        ndarray: The updated solution vector.
    """

    delta_t = T / mt

    # Construct the tridiagonal matrix
    tridiag = tridiag_mat(pde_sol, Boundary_type, t, x, D, L, linearity, T=0.5, mt=1000, mx=100)
    n, _ = pde_sol.shape
    tridiag_identity_matrix = identity(n) - tridiag

    # Compute the right-hand side of the equation
    if linearity == 'linear':
        b = pde_sol.take(time_step, axis=1) + delta_t * source_term(x, t[time_step + 1])
        pde_sol[:, time_step + 1] = apply_bc(spsolve(tridiag_identity_matrix, b), time_step + 1, Boundary_Cond, L, t, Boundary_type)

    elif linearity == 'nonlinear':
        delta_x = L / mx
        r = delta_t / (delta_x**2 * D)
    
        tridiag = tridiag_mat(pde_sol, Boundary_type, t, x, D, L, linearity, T=0.5, mt=1000, mx=100)
        n, _ = pde_sol.shape
        tridiag_identity_matrix = identity(n) - r * tridiag

        b = pde_sol[:, time_step] + delta_t * source_term(pde_sol[:, time_step], x, t[time_step + 1])
        pde_sol[:, time_step + 1] = apply_bc(spsolve(tridiag_identity_matrix, b), time_step + 1, Boundary_Cond, L, t, Boundary_type)

    # Apply boundary conditions to the solution vector
    return pde_sol

def crank_nicholson(pde_sol, t, x, L, Boundary_Cond, Boundary_type, time_step, D, linearity, source_term=None, T=0.5, mt=1000, mx=100, tol=1e-6, max_iter=100):
    """
    Solves a PDE using the Crank-Nicolson method.

    Args:
        pde_sol (ndarray): The solution vector of the PDE.
        t (ndarray): The time vector.
        x (ndarray): The spatial vector.
        L (float): The length of the domain.
        Boundary_Cond (function): The boundary condition function.
        Boundary_type (str): The type of boundary condition.
        time_step (int): The current time step.
        D (function or ndarray): The diffusion coefficient. Can be a function or an ndarray.
        linearity (str): The linearity of the PDE. Can be 'linear' or 'nonlinear'.
        source_term (function or ndarray): The source term. Can be a function or an ndarray. Defaults to None.
        T (float): The final time. Defaults to 0.5.
        mt (int): The number of time steps. Defaults to 1000.
        mx (int): The number of spatial steps. Defaults to 100.
        tol (float): The tolerance for the Newton method. Defaults to 1e-6.
        max_iter (int): The maximum number of iterations for the Newton method. Defaults to 100.

    Returns:
        ndarray: The updated solution vector.
    """

    # Define a function to compute the Jacobian and residual vectors for Newton's method
    def compute_jacobian_and_residual(u, half_tridiag, source_term, x, t, time_step, delta_t):
        """
        Computes the Jacobian matrix and residual vector for Newton's method.

        Args:
            u (ndarray): The solution vector at the current time step.
            half_tridiag (ndarray): Half the tridiagonal matrix.
            source_term (function or ndarray): The source term. Can be a function or an ndarray.
            x (ndarray): The spatial vector.
            t (ndarray): The time vector.
            time_step (int): The current time step.
            delta_t (float): The time step size.

        Returns:
            tuple: A tuple containing the Jacobian matrix and residual vector.
        """
        n = len(u)

        # Compute the residual vector
        F = half_tridiag.dot(u) + delta_t * source_term(x, t[time_step + 1]) - u

        # Compute the Jacobian matrix
        J = half_tridiag - identity(n)

        return J, F

    delta_t = T / mt

    # Construct the tridiagonal matrix
    tridiag = tridiag_mat(pde_sol, Boundary_type, t, x, D, L, linearity, T=0.5, mt=1000, mx=100)
    n, _ = pde_sol.shape
    half_tridiag = 0.5 * tridiag
    Identity_matrix = identity(n)

    # Compute the matrices needed for the CN method
    cn_mat1 = (Identity_matrix + half_tridiag)
    cn_mat2 = (Identity_matrix - half_tridiag)

    # Compute the right-hand side of the equation
    if linearity == 'linear':
        
        b = cn_mat1.dot(pde_sol[:, time_step]) + (delta_t/2) * (source_term(x, t[time_step]) + source_term(x, t[time_step+1]))
        b = apply_bc(b, time_step, Boundary_Cond, L, t, Boundary_type, CN=True)
        pde_sol[:, time_step+1] = spsolve(cn_mat2, b)


    #attempting to use Newton's method to solve the nonlinear problem, as per lecture notes.
    elif linearity == 'nonlinear':
        
        u = pde_sol.take(time_step, axis=1)
        
        for _ in range(max_iter):
            J, F = compute_jacobian_and_residual(u, tridiag, source_term, x, t, time_step, delta_t)
            V = -spsolve(J, F)
            u_new = u + V

            if np.linalg.norm(V) < tol:
                break

            u = u_new

        pde_sol[:, time_step + 1] = apply_bc(u_new, time_step + 1, Boundary_Cond, L, t, Boundary_type)    

    return pde_sol

import numpy as np

def finite_difference(L, T, mx, mt, Boundary_type, Boundary_Cond, Initial_C, discretisation, source_term, D , linearity):
    """
    Solves a partial differential equation using the finite difference method.
    
    Args:
    L (float): length of the spatial domain
    T (float): length of the time domain
    mx (int): number of spatial grid points
    mt (int): number of time grid points
    Boundary_type (str): type of boundary condition ('Dirichlet' or 'Neumann')
    Boundary_Cond (function): function specifying the boundary condition
    Initial_C (function): function specifying the initial condition
    discretisation (str): type of finite difference method to use ('explicit', 'implicit', or 'cn')
    source_term (function): function specifying the source term in the PDE
    D (function or float): function specifying the diffusion coefficient in the PDE, or a constant value if the PDE is linear
    linearity (str): 'linear' or 'nonlinear' depending on whether the PDE is linear or nonlinear
    
    Returns:
    pde_sol (ndarray): solution to the PDE on the spatial and temporal grid
    t (ndarray): array of time values corresponding to the temporal grid
    """
    
    # Calculate spatial and temporal grid spacing
    delta_x = L / mx
    delta_t = T / mt

    # Create arrays of mesh points in space and time
    x = np.arange(0, L+delta_x, delta_x)
    t = np.arange(0, T+delta_t, delta_t)    
    
    if linearity == 'linear':
        # Calculate Euler flag for linear PDE
        euler_flag = D(x)*delta_t/(delta_x**2)


    # Initialise the solution matrix
    pde_sol = np.zeros(shape=(len(x), len(t)))
    
# Check if nonlinear PDE is being solved with the correct method
    if linearity == 'nonlinear' and discretisation not in ['implicit', 'cn']:
        raise ValueError("Nonlinear equations can only be solved using the Implicit Euler and Crank-Nicholson methods.")

    
    # Determine the finite difference method to use
    if discretisation == 'explicit':
        # Check if explicit euler method is stable
        if (euler_flag > 0.5).any():
            raise ValueError('Euler flag greater than 0.5, Explicit euler will break down.')
        discretisation = explicit_euler
    elif discretisation == 'implicit':
        discretisation = implicit_euler
    elif discretisation == 'cn':
        discretisation = crank_nicholson
    else:
        raise ValueError('Please choose from Explicit Euler,Implicit Euler,CN')

    # Get initial conditions and apply
    pde_sol[:, 0] = np.vectorize(Initial_C)(x, L)
    
    # Solve PDE for each time step
    for time_step in range(0, mt):
        # Carry out solver step, including the boundaries
        pde_sol = discretisation(pde_sol, t, x, L, Boundary_Cond, Boundary_type, time_step, D,linearity, source_term)

    return pde_sol, t



