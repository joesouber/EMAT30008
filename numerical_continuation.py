#%%
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

#%%



def npc(f, u0, pars, vary_min, discretisation='shooting', solver=fsolve, pc='none'):
    """
    Perform numerical path continuation on a system of ODEs.

    Parameters:
    -----------
    f: callable
        The system of ODEs.
    u0: array-like
        The initial conditions for the ODEs.
    pars: array-like
        The parameters of the ODE system.
    vary_min: float
        The minimum value of the varying parameter.
    discretisation: str, optional
        The discretization method for the continuation (default is 'shooting').
    solver: callable, optional
        The solver for the continuation (default is fsolve).
    pc: callable or str, optional
        The function for parameter continuation (default is 'none').

    Returns:
    --------
    parameter_array: numpy array
        Array of parameter values for the continuation.
    sol_list: numpy array
        Array of solution points for the continuation.
    """

    # Generate an array of parameter values to use in the continuation.
    parameter_array = np.linspace(pars[0], vary_min, 30)

    # Initialize an empty list to store the solutions.
    sol_list = []

    # Iterate over the parameter values.
    for i in parameter_array:
        # Update the parameter value in `pars`.
        pars.__setitem__(0, i)

        # Set the initial conditions for the ODE system.
        initial_pars0 = (pc or '') and ((pc or '') != 'none') and (pc, pars) or pars

        # Solve the ODE system using the specified method and solver.
        sol = np.array(solver(discretisation(f), u0, args=initial_pars0))

        # Append the solution to the list of solutions.
        sol_list.append(sol)

        # Set the initial conditions for the next step to be the current solution.
        u0 = sol

    # Return the parameter values and the list of solutions as a tuple.
    return parameter_array, np.array(sol_list)


def pseudo_arclength(ode_func, init_conds, param_bounds, step_size, discretisation):
    """
    Perform pseudo-arclength continuation on a system of ODEs.

    Parameters:
    -----------
    ode_func: callable
        The system of ODEs.
    init_conds: array-like
        The initial conditions for the ODEs.
    param_bounds: tuple
        The starting and ending values for the parameter.
    step_size: float
        The step size for the continuation.
    discretisation: callable
        The discretisation function for the continuation.

    Returns:
    --------
    param_list: list
        List of parameter values for the continuation.
    solution_points: list
        List of solution points for the continuation.
    """

    param_start, param_end = param_bounds
    param_list = [param_start, param_start + step_size]

    # Create a dictionary to store lambda functions and nested functions
    function_map = {
        'should_stop': lambda K: (param_end > param_start and K > param_end) or (param_end < param_start and K < param_end),
        'first_solution': lambda U: discretisation(U, ode_func, param_start),
        'second_solution': lambda U: discretisation(U, ode_func, param_list[1]),
        'arclength_constraint': lambda next_step, prediction, delta: np.dot(next_step - prediction, delta),
        'combined_equations': lambda next_step: np.append(discretisation(next_step[1:], ode_func, next_step[0]), function_map['arclength_constraint'](next_step, prediction, delta)),
    }

    # Find the first two solution points using fsolve
    first_point = fsolve(function_map['first_solution'], init_conds)
    second_point = fsolve(function_map['second_solution'], first_point)

    solution_points = [first_point, second_point]

    # Loop until the stopping condition is met
    while True:
        # Calculate the previous and current points and their difference (delta)
        previous = np.concatenate(([param_list[-2]], solution_points[-2]))
        current = np.concatenate(([param_list[-1]], solution_points[-1]))
        delta = current - previous

        # Predict the next solution point and solve the combined_equations
        prediction = current + delta
        next_solution = fsolve(function_map['combined_equations'], prediction)

        # Append the new solution point and parameter value
        solution_points.append(next_solution[1:])
        param_list.append(next_solution[0])

        # Check if the stopping condition is met, and break the loop if it is
        if function_map['should_stop'](param_list[-1]):
            break

    return param_list, solution_points





#
# %%
