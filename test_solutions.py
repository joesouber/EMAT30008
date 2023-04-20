#%%
import math
import numpy as np
from ODE import solve_ode
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from shooting_plot import shooting_generalised, plot_hopf_shooting,plot_solutions
from scipy.integrate import odeint

#%%
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
#%%
def test_shooting_inputs():
    def predator_prey(t, y, args = [1, 0.1, 0.1]):
        x = y[0]
        y = y[1]

        a = args[0]
        d = args[1]
        b = args[2]
        
        dxdt = x * (1 - x) - (a * x * y) / (d + x)
        dydt = b * y * (1 - (y / x))
        return np.array([dxdt, dydt])
    
    def pc_predator_prey(u0, args = [1, 0.1, 0.1]):
        return predator_prey(0, u0, args = [1, 0.1, 0.1])[0]

# Define the initial guess for the shooting method
    u0 = [0.8, 0.3,30]
    failed_tests = []

    # Test with correct inputs
    try:
        plot_solutions(predator_prey, pc_predator_prey, u0)
    except Exception as e:
        failed_tests.append("Test with correct inputs passed")

    # Test with incorrect ODE output
    def incorrect_ode_output(t, y, args=[1, 0.1, 0.1]):
        return y
    try:
        plot_solutions(incorrect_ode_output, pc_predator_prey, u0)
        failed_tests.append("Test with incorrect ODE output did not raise an exception.")
    except AssertionError:
        pass
    except Exception as e:
        failed_tests.append("Test with incorrect ODE output raised an  exception")

    # Test with incorrect phase condition output
    def incorrect_pc_output(u0, args=[1, 0.1, 0.1]):
        return np.array([1, 2])
    try:
        plot_solutions(predator_prey, incorrect_pc_output, u0)
        failed_tests.append("Test with incorrect phase condition output did not raise an exception.")
    except AssertionError:
        pass
    except Exception as e:
        failed_tests.append("Test with incorrect phase condition output raised an  exception ")

    # Test with incorrect u0 size
    u0_incorrect_size = [0.8, 0.3]
    try:
        plot_solutions(predator_prey, pc_predator_prey, u0_incorrect_size)
        failed_tests.append("Test with incorrect u0 size did not raise an exception.")
    except AssertionError:
        pass
    except Exception as e:
        failed_tests.append("Test with incorrect u0 size raised an  exception ")

    # Test with incorrect step size type
    max_step_incorrect_type = '0.01'
    try:
        plot_solutions(predator_prey, pc_predator_prey, u0, max_step=max_step_incorrect_type)
        failed_tests.append("Test with incorrect step size type did not raise an exception.")
    except AssertionError:
        pass
    except Exception as e:
        failed_tests.append("Test with incorrect step size type raised an  exception ")

    if len(failed_tests) == 0:
        print("All tests passed!")
    else:
        print('Some input tests failed:')
        for test in failed_tests:
            print(test)

import numpy as np
from scipy.integrate import solve_ivp
#%%

def test_shooting_generalised():
    
    def normal_hopf(u0, t, beta):

    #beta = pars[0]
        u1, u2 = u0[0], u0[1]

        du1dt = beta * u1 - u2 - (u1) * (u1 ** 2 + u2 ** 2)
        du2dt = u1 + beta * u2 - (u2) * (u1 ** 2 + u2 ** 2)
        return np.array([du1dt, du2dt])


# phase condition for the normal Hopf bifurcation (u1 gradient = 0)
    def pc_normal_hopf(u0, pars):
        return normal_hopf(u0, 0, pars)[0]

# Solve the ODE system using the specified method and solver.
    
    failed_tests = []
    u0_guess_hopfnormal = np.array([1.5, 0.1, 6.1])
    pc = pc_normal_hopf
    beta = 2
    initial_pars0 = (pc, beta)

    # Test for incorrect ODE output and size
    def f1(u0, t, beta):
        return np.array([beta, beta,beta])
    
    try:
        sol = np.array(fsolve(shooting_generalised(f1), u0_guess_hopfnormal, args=initial_pars0))
        failed_tests.append("Test 1 failed: incorrect ODE output and size did not raise an exception")
    except:
        pass

    # Test for incorrect phase condition output and shape
    def pc_wrong_output(u0, pars):
        return np.array([1, 2, 3])

    try:
        sol = np.array(fsolve(shooting_generalised(normal_hopf), u0_guess_hopfnormal, args=(pc_wrong_output, beta)))
        failed_tests.append("Test 2 failed: incorrect phase condition output and shape did not raise an exception")
    except:
        pass

    # Test for incorrect u0 size
    u0_guess_hopfnormal_wrong = np.array([1.5, 0.1])
    
    try:
        sol = np.array(fsolve(shooting_generalised(normal_hopf), u0_guess_hopfnormal_wrong, args=initial_pars0))
        failed_tests.append("Test 3 failed: incorrect u0 size did not raise an exception")
    except:
        pass

    # Test for incorrect step size type
    def f2(u0, t, beta):
        return np.array([beta,'bad'])
    
    try:
        sol = np.array(fsolve(shooting_generalised(f2), u0_guess_hopfnormal, args=initial_pars0))
        failed_tests.append("Test 4 failed: incorrect step size type did not raise an exception")
    except:
        pass


    if len(failed_tests) > 0:
        print("Some tests failed:")
        for failed_test in failed_tests:
            print(failed_test)
    else:
        print("All tests passed!")







#%%


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
    test_shooting_inputs()
    test_shooting_generalised()
# %%
