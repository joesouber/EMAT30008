#%%
import unittest
from ODE import solve_ode
from ODE_PDE_funcs import f_true_solution, f
import numpy as np
from numerical_shooting import shooting
from scipy.optimize import fsolve
#%%


class TestODESolver(unittest.TestCase):
    
    def test_euler_solver(self):
        # Test the Euler method solver on a simple ODE
        x0 = 1
        t0 = 0
        t1 = 1
        h = 0.1
        X, T = solve_ode(f, x0, t0, t1, 'euler', h, False)
        self.assertTrue(np.abs(X[-1]-f_true_solution(T)) < 10**-3)
    
    def test_RK4_solver(self):
        # Test the Runge-Kutta 4 method solver on a simple ODE
        x0 = 1
        t0 = 0
        t1 = 1
        h = 0.1
        X, T = solve_ode(f, x0, t0, t1, 'RK4', h, False)
        self.assertTrue(np.abs(X[-1]-f_true_solution(T)) < 10**-8)
    
    def test_heun_solver(self):
        # Test the Heun method solver on a simple ODE
        x0 = 1
        t0 = 0
        t1 = 1
        h = 0.1
        X, T = solve_ode(f, x0, t0, t1, 'heun', h, False)
        self.assertTrue(np.abs(X[-1]-f_true_solution(T)) < 10**-8)
        

    def calculate_error(f, true_sol, x0, t0, t1, dt_list, *args):
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
            X_true = true_sol(T)

            # Compute the errors
            errors['euler'].append(np.abs(X_euler - X_true)[-1])
            errors['RK4'].append(np.abs(X_RK4 - X_true)[-1])
            errors['heun'].append(np.abs(X_heun - X_true)[-1])

        return errors






    


#%% Shooting Test
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from numerical_shooting import shooting
import numpy as np


# ode to solve using shooting
def hopf_normal(t, u):

    u1, u2 = u
    beta = 1
    sigma = -1

    du1dt = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    du2dt = u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)

    return np.array((du1dt, du2dt))



# exact solution to differential equation
def true_hopf_sol(t):
    beta = 1
    theta = 0.0

    u1 = np.sqrt(beta) * np.cos(t + theta)
    u2 = np.sqrt(beta) * np.sin(t + theta)

    return np.array([u1, u2])



# phase condition function for the hopf
def pc_hopf(u0, *args):
    return hopf_normal(0, u0, *args)[0]


def test_limit_cycle(test_f, true_f, U0, pc, tolerance, endpoints,*args):
    sol_shooting = shooting(test_f, U0, pc,fsolve, *args)
    u0 = sol_shooting[:-1]
    T = sol_shooting[-1]

    # Use solve_ivp to get limit cycle values
    sol = solve_ivp(test_f,(0,T),u0)
    t_sol = sol.t
    u_sol = sol.y

    # transpose array to fit true_f dimensions
    u_sol = u_sol.T

    # get array shape to create according zero array
    rows, columns = u_sol.shape
    exact_sol = np.zeros([rows, columns])

    # get true solutions to the ode for the same times.
    for i in range(0,len(t_sol)):
        exact_sol[i] = true_f(t_sol[i])


    if endpoints == False:
        # compare true solutions to solutions obtained using the limit cycle.
        return np.allclose(u_sol, exact_sol, tolerance)
    elif endpoints == True:
        # compare initial conditions with limit cycle endpoints
        return np.allclose(exact_sol[0,:], exact_sol[-1,:], tolerance)
    


def test_limit_cycle_hopf(tol_range=np.logspace(-1,-21,21), init_guess=(-1,0,6)):
    """
    Test whether the found limit cycle of the Hopf normal form ode
    matches the true values at the same time points. This function
    returns the tolerance level at which the limit cycle was found
    to be accurate for the given initial guesses.
    """
    # Define parameters for the limit cycle test
    tolerance = tol_range  # range of tolerances to test for
    U0_2d = init_guess  # initial condition estimates
    
    # Test found limit cycle matches true values at same time points for 2D ode
    for i in range(0,len(tolerance)):
        passed = test_limit_cycle(hopf_normal, true_hopf_sol, U0_2d, pc_hopf, tolerance[i], False)
        if passed == False:
            return f"Limit cycle was found to be accurate at a tolerance level of {tolerance[i-1]} for a 2-dimensional ODE."
        elif i == len(tolerance):
            return f"Limit cycle was found to be accurate at a tolerance level of {tolerance[-1]} or greater, for a 2-dimensional ODE."

# %%
test_limit_cycle_hopf(tol_range=np.logspace(-1,-21,21), init_guess=(-1,0,6))
# %%
