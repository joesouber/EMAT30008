from matplotlib import pyplot as plt
import numpy as np
from ODE import solve_ode

def plot_error(f, f_true_solution, x0, t0, t1, dt_list, *args):
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
    def calculate_error():
        """
        Calculate the error between the approximate solution obtained using Euler, RK4 or Heun method and the true solution
        for different timesteps.

        Parameters:
        f (function): Function that returns the derivative of the function to be solved.
        true_sol (function): Function that returns the true solution of the ODE.
        x0 (float): Initial value of the function.
        t0 (float): Initial time.
        t1 (float): Final time.
        dt_list (list): List of timestep sizes to compare.
        args (tuple): Tuple of additional arguments for f.

        Returns:
        dict: Dictionary containing the errors for each method.
        """
        # Define the approximate methods to compare
        methods = ['euler', 'RK4', 'heun']

        # Define the dictionary to store the errors
        errors = {}
        for method in methods:
            errors[method] = []

        # Loop over the timestep sizes
        for dt in dt_list:
            # Compute the approximate solutions
            X_euler = solve_ode(f, x0, np.arange(t0, t1 + dt, dt), dt, 'euler', False, *args)[-1]
            X_RK4 = solve_ode(f, x0, np.arange(t0, t1 + dt, dt), dt, 'RK4', False, *args)[-1]
            X_heun = solve_ode(f, x0, np.arange(t0, t1 + dt, dt), dt, 'heun', False, *args)[-1]

            # Compute the true solution
            T = np.arange(t0, t1 + dt, dt)
            X_true = f_true_solution(T)

            # Compute the errors
            errors['euler'].append(np.abs(X_euler - X_true)[-1])
            errors['RK4'].append(np.abs(X_RK4 - X_true)[-1])
            errors['heun'].append(np.abs(X_heun - X_true)[-1])

        return errors

    # Calculate the errors for the three methods using the given parameters
    errors = calculate_error()

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
    