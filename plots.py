from matplotlib import pyplot as plt
import numpy as np
from ODE_PDE_funcs import f, f_true_solution
from Numerical_solvers_tests import calculate_error

def plot_error(f_test, f_true_solution, x0, t0, t1, dt_list, *args):
    """
    Plots the error for the Euler, RK4, and Heun methods for different timesteps.

    Parameters:
    -----------
    f : callable
        The function defining the ODE system, must take inputs (x, t, *args).
    true_sol : callable
        A function that returns the true solution to the ODE system, must take inputs (t, x0, *args).
    x0 : float or list
        The initial condition(s) for the ODE system.
    t0 : float
        The starting time for the solution.
    t1 : float
        The end time for the solution.
    dt_list : list
        A list of timesteps to use for the numerical solutions.
    *args : tuple, optional
        Extra arguments to be passed to `f` and `true_sol`.

    """
    # Calculate the errors for the three methods using the given parameters
    errors = calculate_error(f_test, f_true_solution, x0, t0, t1, dt_list, *args)

    # Plot the errors for the three methods
    plt.figure()
    plt.loglog(dt_list, errors['euler'],'o-', label='Euler')
    plt.loglog(dt_list, errors['RK4'],'o-', label='RK4')
    plt.loglog(dt_list,errors['heun'],'o-',label='Heun')
    plt.xlabel('Timestep')
    plt.ylabel('Error')
    plt.title('Errors for Numerical Methods')
    plt.legend()
    plt.show()
    