#%%
import math
import numpy as np
from ODE import solve_ode

def test_solution(tolerance):
    # Define the function to be solved
    def f(x, t):
        return x

    # Define the true solution of the ODE
    def true_sol(t):
        return np.exp(t)

    # Set up the initial conditions
    x0 = 1.0
    t_eval = np.linspace(0, 5, 101) # time points to evaluate the solution
    max_step = 0.01 # maximum step size for the solver
    methods = ['euler', 'heun', 'rk4'] # numerical methods for the solver
    system = False # the ODE is not a system

    # Iterate over the methods and test each one
    for method in methods:
        # Solve the ODE using the solve_ode function
        X = solve_ode(f, x0, t_eval, max_step, method, system)

        # Test the numerical solution against the true solution using allclose
         # set a tolerance value
        is_close = np.allclose(X[:,0], true_sol(t_eval), rtol=tolerance, atol=tolerance)
        print(f"{method.upper()}: Numerical and exact solutions match closely: {is_close}")

if __name__ == '__main__':
    test_solution(tolerance=1)

# %%
