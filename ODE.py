#%%
import matplotlib.pyplot as plt
import math
import numpy as np
from checks import *


def euler_step(f, x0, t0, max_step, *pars):
    """
    Performs one step of the Euler method for solving ODEs.
    
    Args:
        f (callable): The function representing the ODE.
        x0 (float): The initial value of x.
        t0 (float): The initial value of t.
        max_step (float): The step size for the Euler method.
        *pars: Additional parameters to be passed to the function f.
    
    Returns:
        tuple: A tuple containing the updated values of x and t.
    """

    check_callable(f)
    check_pars(pars)

    x1 = x0 + max_step * f(x0, t0, *pars)
    t1 = t0 + max_step
    return x1, t1

def RK4_step(f, x0, t0, max_step, *pars):
    """
    Performs one step of the fourth-order Runge-Kutta (RK4) method for solving ODEs.
    
    Args:
        f (callable): The function representing the ODE.
        x0 (float): The initial value of x.
        t0 (float): The initial value of t.
        max_step (float): The step size for the RK4 method.
        *pars: Additional parameters to be passed to the function f.
    
    Returns:
        tuple: A tuple containing the updated values of x and t.
    """

    check_callable(f)
    check_pars(pars)
    
    # Compute the intermediate values k1, k2, k3, and k4
    k1 = f(x0, t0, *pars)
    k2 = f(x0 + max_step * 0.5 * k1, t0 + 0.5 * max_step, *pars)
    k3 = f(x0 + max_step * 0.5 * k2, t0 + 0.5 * max_step, *pars)
    k4 = f(x0 + max_step * k3, t0 + max_step, *pars)

    # Combine the intermediate values to compute the total increment k
    k = 1 / 6 * max_step * (k1 + 2 * k2 + 2 * k3 + k4)

    x1 = x0 + k
    t1 = t0 + max_step

    return x1, t1



def heun_step(f, x0, t0, max_step, *pars):

    check_callable(f)
    #check_numeric(x0, max_step)
    check_pars(pars)

    k1 = f(x0, t0, *pars)
    x1_tilde = x0 + max_step * k1
    k2 = f(x1_tilde, t0 + max_step, *pars)

    k = 1 / 2 * max_step * (k1 + k2)
    x1 = x0 + k
    t1 = t0 + max_step

    return x1, t1

def solve_to(f, x0, t0, t1, max_step, method='RK4', *pars):
    """
    Solve an ordinary differential equation (ODE) by iterating the specified numerical method until the specified end.
    
    Args:
        f: function defining the ODE, takes arguments (t, x, *pars) where t is the current time, x is the current state,
            and *pars are additional parameters
        x0: initial state
        t0: initial time
        t1: end time
        max_step: maximum step size allowed, used to limit step size to avoid overshooting t1
        method: string specifying the numerical method to use, options are 'euler' (or 'Euler') for the Euler method,
            'RK4' (or 'rk4', or 'Runge-Kutta') for the 4th-order Runge-Kutta method, and 'heun' (or 'Heun', or 'h', or
            'Hun') for the Heun method
        *pars: additional parameters to pass to the numerical method
        
    Returns:
        x0: final state after the ODE has been solved
    """
    
    # Loop until we reach t1
    while t0 < t1:
        # Limit step size to avoid overshooting t1
        max_step = min(max_step, t1 - t0)
        
        # Take a step using the specified numerical method
        if method in ('euler', 'Euler'):
            x0, t0 = euler_step(f, x0, t0, max_step, *pars)
        elif method in ('RK4', 'rk4', 'Runge-Kutta'):
            x0, t0 = RK4_step(f, x0, t0, max_step, *pars)
        elif method in ('heun', 'Heun', 'h', 'Hun'):
            x0, t0 = heun_step(f, x0, t0, max_step, *pars)
    
    # Return the final state
    return x0





def solve_ode(f, x0, t_eval, max_step, method, system, *pars):
    """
    Solve an ordinary differential equation (ODE) by iterating the specified numerical method until the specified end.
    
    Args:
        f: function defining the ODE, takes arguments (t, x, *pars) where t is the current time, x is the current state,
            and *pars are additional parameters
        x0: initial state, either a scalar if the ODE is not a system, or a numpy array of length n if the ODE is a
            system of n ODEs
        t_eval: list of evaluation times
        max_step: maximum step size allowed, used to limit step size to avoid overshooting t_eval
        method: string specifying the numerical method to use, options are 'euler' (or 'Euler') for the Euler method,
            'RK4' (or 'rk4', or 'Runge-Kutta') for the 4th-order Runge-Kutta method, and 'heun' (or 'Heun', or 'h', or
            'Hun') for the Heun method
        system: boolean indicating whether the ODE is a system of ODEs or not
        *pars: additional parameters to pass to the numerical method
        
    Returns:
        X: numpy array containing the solution at the specified evaluation times. If the ODE is a system of n ODEs, X
            has shape (n, len(t_eval)), otherwise X has shape (len(t_eval), 1)
    """
    
    # Define the empty X array depending on size of x0 and t_eval
    if system:
        X = np.zeros((len(t_eval), len(x0)))
    else:
        X = np.zeros((len(t_eval), 1))
    X[0] = x0
    
    # Loop over evaluation times and solve the ODE
    steps = len(t_eval) - 1
    for i in range(steps):
        T_i = t_eval[i]
        T_ip1 = t_eval[i + 1]
        X_solved = solve_to(f, X[i], T_i, T_ip1, max_step, method, *pars)
        X[i+1] = X_solved
    
    # If it's a system of ODEs, transpose X before returning
    if system:
        X = X.transpose()
    
    return X


