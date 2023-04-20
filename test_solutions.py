#%%
import math
import numpy as np
from ODE import solve_ode
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from shooting_plot import shooting_generalised, plot_hopf_shooting
from scipy.integrate import odeint
#ode
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

#shooting
def test_shooting(tolerance):
    def normal_hopf(u0, t, beta):

    
        u1, u2 = u0[0], u0[1]

        du1dt = beta * u1 - u2 - (u1) * (u1 ** 2 + u2 ** 2)
        du2dt = u1 + beta * u2 - (u2) * (u1 ** 2 + u2 ** 2)
        return np.array([du1dt, du2dt])


#phase condition for the normal Hopf bifurcation (u1 gradient = 0)
    def pc_normal_hopf(u0, pars):
        return normal_hopf(u0, 0, pars)[0]

    u0_guess_hopfnormal = np.array([1.5, 0.1, 6.1])
    pc = pc_normal_hopf
    beta = 2
    initial_pars0 = (pc, beta)
    # Solve the ODE system using the specified method and solver.
    sol = np.array(fsolve(shooting_generalised(normal_hopf),u0_guess_hopfnormal, args=initial_pars0))
    
    
    t = np.linspace(0, sol[-1], 2000)
    sol1 = odeint(normal_hopf, sol[:-1], t, args=(beta,))
    
    # Compute the phase angle.
    theta = pc(sol[:-1], beta)
    
    # Compute the analytical solutions.
    u1_exact = np.sqrt(beta) * np.cos(t + theta)
    u2_exact= np.sqrt(beta) * np.sin(t + theta)

    u1_num = sol1[:,0]
    u2_num = sol1[:,1]

    assert np.allclose(u1_num, u1_exact, rtol=tolerance, atol=tolerance),"Error: Numerical and exact solutions do not match closely."
    assert np.allclose(u2_num, u2_exact, rtol=tolerance, atol=tolerance),"Error: Numerical and exact solutions do not match closely."
    
    print('Shooting method: Numerical and exact solutions match closely to the specified tolerance.')


if __name__ == '__main__':
    test_solution(tolerance=1)
    test_shooting(tolerance=1e-3)
# %%
