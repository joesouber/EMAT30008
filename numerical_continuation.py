#%%
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

#%%



def npc(f, u0, pars, vary_min, discretisation='shooting',solver=fsolve, pc='none'):
    
    # Generate an array of parameter values to use in the continuation. Done back-to-front of sorts. 
    #step set to 50, can reduce this to reduce time, however reduces resolution.
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

def pseudo_arclength(f, initial_conds, parameters, param_step, discretisation):


    # Unpack parameters and find the second alpha value, and the stopping condition (alpha_end is reached)
    start_value, end_value = parameters
    value_list = [start_value, start_value + param_step]

    # Determine the stopping condition function based on the parameter range
    if end_value > start_value:
        def end_init(K):
            return K > end_value
    else:
        def end_init(K):
            return K < end_value


    def func_u0(U, f):
        return discretisation(U, f, start_value)

    u0 = fsolve(func_u0, initial_conds, args=(f,))
    
    def func_u1(U,f):
        return discretisation(U, f, value_list[1])
    
    u1 = fsolve(func_u1,u0,args=(f,))


    

    # Solve until the final alpha value is reached using pseudo-arclength continuation
    U_list = [u0, u1]
    for i in range(2, 10000):
    # Check if the stopping condition has been reached
        if end_init(value_list[-1]):
            break

    # Generate the secant
        nu0 = np.concatenate(([value_list[i-2]], U_list[i-2]))
        nu1 = np.concatenate(([value_list[i-1]], U_list[i-1]))
        delta = nu1 - nu0

        # Predict solution
        nu2_ = nu1 + delta
        
        def pseudo_arc_equation(nu2):
            return np.dot(nu2 - nu2_, delta)

# Define the function to pass to fsolve
        def func(nu2, f):
            return np.append(discretisation(nu2[1:], f, nu2[0]), pseudo_arc_equation(nu2))

# Solve the function
        sol = fsolve(func, nu2_, args=(f,))
        U_list.append(sol[1:])
        value_list.append(sol[0])

    return value_list, U_list


#
# %%
