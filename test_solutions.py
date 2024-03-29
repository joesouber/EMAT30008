#%%
import math
import numpy as np
from ODE import solve_ode
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from shooting_plot import shooting_generalised, plot_hopf_shooting,plot_solutions
from scipy.integrate import odeint
from numerical_continuation import npc, pseudo_arclength
from PDE import *
#%%
#ode
def test_ODE_inputs():
    failed_tests = []
    def f_ddot(X,t):
        x, y = X
        dxdt = y
        dydt = -x
        return np.array([dxdt, dydt])

    # Test 1: Correct input
    try:
        x0 = np.array([1, 0])  # Initial conditions
        t_eval = np.linspace(0, 10, 101)  # Evaluation points
        max_step = 0.1  # Maximum step size
        method = 'RK4'  # Numerical method
        system = True  # System of ODEs
        X = solve_ode(f_ddot, x0, t_eval, max_step, method, system)
        assert X.shape == (2, 101)
    except:
        failed_tests.append("Test 1 failed ")
    else:
        print("Test 1 passed")


    # Test 2: Wrong type of function
    try:
        x0 = np.array([1, 0])  # Initial conditions
        t_eval = np.linspace(0, 10, 101)  # Evaluation points
        max_step = 0.1  # Maximum step size
        method = 'RK4'  # Numerical method
        system = True  # System of ODEs
        X = solve_ode("f_ddot", x0, t_eval, max_step, method, system)
        assert False
    except TypeError:
        failed_tests.append("Test 2 passed")


    # Test 3: ODE has incorrect output
    def f_wrong_output(x, t):
        return x

    try:
        x0 = np.array([1, 0])  # Initial conditions
        t_eval = np.linspace(0, 10, 101)  # Evaluation points
        max_step = 0.1  # Maximum step size
        method = 'RK4'  # Numerical method
        system = True  # System of ODEs
        X = solve_ode(f_wrong_output, x0, t_eval, max_step, method, system)
        assert False
    except AssertionError:
        failed_tests.append("Test 3 passed")


    # Test 4: ODE outputs wrong size
    def f_wrong_size(x, t):
        return np.array([x[0]])

    try:
        x0 = np.array([1, 0])  # Initial conditions
        t_eval = np.linspace(0, 10, 101)  # Evaluation points
        max_step = 0.1  # Maximum step size
        method = 'RK4'  # Numerical method
        system = True  # System of ODEs
        X = solve_ode(f_wrong_size, x0, t_eval, max_step, method, system)
        assert False
    except AssertionError:
        failed_tests.append("Test 4 passed")


    # Test 5: x0 is wrong type and size
    try:
        x0 = 1  # Initial conditions
        t_eval = np.linspace(0, 10, 101)  # Evaluation points
        max_step = 0.1  # Maximum step size
        method = 'RK4'  # Numerical method
        system = True  # System of ODEs
        X = solve_ode(f_ddot, x0, t_eval, max_step, method, system)
        assert False
    except TypeError:
        failed_tests.append("Test 5 passed")


    # Test 6: t_eval is wrong type and size
    try:

        x0 = np.array([1, 0])  # Initial conditions
        t_eval = 10  # Evaluation points
        max_step = 0.1  # Maximum step size
        method = 'RK4'  # Numerical method
        system = True  # System of ODEs
        X = solve_ode(f_ddot, x0, t_eval, max_step, method, system)
        assert False
    except TypeError:
        failed_tests.append("Test 6 passed")

    
    if len(failed_tests) == 0:
        print('\n---------------------------------------\n')
        print("All tests passed!")
        print('\n---------------------------------------\n')
    else:
        print('Some input tests failed:')
        for test in failed_tests:
            print('\n---------------------------------------\n')
            print(test)
            print('\n---------------------------------------\n')

def test_ode_output(tolerance):
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
        print('\n---------------------------------------\n')
        print(f"{method.upper()}: Numerical and exact solutions match closely: {is_close}")
        print('\n---------------------------------------\n')

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
        return [y,'bad']
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
        print('\n---------------------------------------\n')
        print("All tests passed!")
        print('\n---------------------------------------\n')
    else:
        print('Some input tests failed:')
        for test in failed_tests:
            print('\n---------------------------------------\n')
            print(test)
            print('\n---------------------------------------\n')



def test_shooting_generalised_inputs():
    
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
        print('\n---------------------------------------\n')
        print("Some tests failed:")
        print('\n---------------------------------------\n')
        for failed_test in failed_tests:
            print('\n---------------------------------------\n')
            print(failed_test)
            print('\n---------------------------------------\n')
    else:
        print('\n---------------------------------------\n')
        print("All tests passed!")
        print('\n---------------------------------------\n')


def test_shooting_output(tolerance):
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
    print('\n---------------------------------------\n')
    print('Shooting method: Numerical and exact solutions match closely to the specified tolerance.')
    print('\n---------------------------------------\n')

#%% Continuation ##WIP
def continuation_test():
    failed_tests  = []
    def cubic(x, pars):
        """
        This function defines a cubic equation
        :param x: Value of x
        :param pars: Defines the additional parameter c
        :return: returns the value of the cubic equation at x
        """
        c = pars[0]
        return x ** 3 - x + c
    
    #Test 1: incorrect function type
    try:

        u0_guess_cubic = np.array([1])
        np_par_list, np_sol_list = npc('cubic', u0_guess_cubic, [-2], 2, lambda x: x, fsolve,pc='none')
        assert False
    except TypeError:
        failed_tests.append("Test 1 passed")

    #Test 2: incorrect function output
    def cubic_wrong_output(x, pars):
        return np.array([1, 2, 3])
    try:
        u0_guess_cubic = np.array([1])
        np_par_list, np_sol_list = npc(cubic_wrong_output, u0_guess_cubic, [-2], 2, lambda x: x, fsolve,pc='none')
        assert False
    except ValueError:
        failed_tests.append("Test 2 passed")

    #Test 3: incorrect u0 size
    def cubic_wrong_u0(x, pars):
        return x ** 3 - x + pars[0]
    try:
        u0_guess_cubic = np.array([1, 2])
        np_par_list, np_sol_list = npc(cubic_wrong_u0, u0_guess_cubic, [-2], 2, lambda x: x, fsolve,pc='none')
        assert False
    except ValueError:
        failed_tests.append("Test 3 passed")

    #Test 4: incorrect step size type
    def cubic_wrong_step(x, pars):
        return np.array([1, 'bad'])
    try:
        u0_guess_cubic = np.array([1])
        np_par_list, np_sol_list = npc(cubic_wrong_step, u0_guess_cubic, [-2], 2, lambda x: x, fsolve,pc='none')
        assert False
    except TypeError:
        failed_tests.append("Test 4 passed")

    #Test 5: incorrect step size value
    def cubic_wrong_step_value(x, pars):
        return np.array([1, 0])
    try:
        u0_guess_cubic = np.array([1])
        np_par_list, np_sol_list = npc(cubic_wrong_step_value, u0_guess_cubic, [-2], 2, lambda x: x, fsolve,pc='none')
        assert False
    except ValueError:
        failed_tests.append("Test 5 passed")

    #Test 6: incorrect solver type
    def cubic_wrong_solver(x, pars):
        return np.array([1, 2])
    try:
        u0_guess_cubic = np.array([1])
        np_par_list, np_sol_list = npc(cubic_wrong_solver, u0_guess_cubic, [-2], 2, lambda x: x, 'bad',pc='none')
        assert False
    except TypeError:
        failed_tests.append("Test 6 passed")


    #Test 7: incorrect pc type
    def cubic_wrong_pc(x, pars):
        return np.array([1, 2])
    try:
        u0_guess_cubic = np.array([1])
        np_par_list, np_sol_list = npc(cubic_wrong_pc, u0_guess_cubic, [-2], 2, lambda x: x, fsolve,pc=1)
        assert False
    except TypeError:
        failed_tests.append("Test 8 passed")

    #Test 8: incorrect pc value
    def cubic_wrong_pc_value(x, pars):
        return np.array([1, 2])
    try:
        u0_guess_cubic = np.array([1])
        np_par_list, np_sol_list = npc(cubic_wrong_pc_value, u0_guess_cubic, [-2], 2, lambda x: x, fsolve,pc='bad')
        assert False
    except ValueError:
        failed_tests.append("Test 9 passed")

    

        print('All input tests passed:')
        for test in failed_tests:
            print('\n---------------------------------------\n')
            print(test)
            print('\n---------------------------------------\n')


#PDE

def test_PDE_inputs():


    #explicit euler and dirichlet
    failed_tests  = []
    L = 1 # length of the domain
    T = 0.5 # total time
    mx = 100 # number of grid points in space
    mt = 1000 # number of grid points in time

    def u_exact(x, t,D,L):  
        y = np.exp(-D*(np.pi**2/L**2)*t)*np.sin(np.pi*x/L)
        return y

    # Dirichlet boundary conditions, 0 on both ends.
    def dirichlet_0(x, t):
        return 0

    # Initial condition, sin(pi*x/L).
    def Initial_Condition(x, L):
        y = (np.sin(np.pi*x/L))
        return y

    # Diffusion coefficient,needs to be a function of x, to be compatible with the finite difference method.
    def D(x):
        return x / (x * 10)

    # Source term, needs to be a function of x and t, to be compatible with the finite difference method. 0 for this demonstration.
    def source_term(x,t):
        return 0


    #explicit euler and dirichlet
    u, t = finite_difference(L, T, mx, mt, 'dirichlet', dirichlet_0, Initial_Condition, discretisation='explicit',source_term = source_term, D = D, linearity='linear')
        
    #Test 1: incorrect boundary condition
    try:

        u, t = finite_difference(L, T, mx, mt, 'ABC', dirichlet_0, Initial_Condition, 
                                 discretisation='explicit',source_term = source_term, D = D, linearity='linear')

        assert False
    except AssertionError:
        failed_tests.append("Test 1 passed")

    #Test 2: incorrect initial condition
    try:
            
        u, t = finite_difference(L, T, mx, mt, 1, dirichlet_0, 'Initial_Condition', 
                                discretisation='explicit',source_term = source_term, D = D, linearity='linear')
    
        assert False
    except TypeError:
        failed_tests.append("Test 2 passed")

    #Test 3: incorrect discretisatin already tested within function itself.

    #Test 4: incorrect source term
    try:
                    
            u, t = finite_difference(L, T, mx, mt, 'dirichlet', dirichlet_0, Initial_Condition, 
                                        discretisation='explicit',source_term = 1, D = D, linearity='linear')
            
            assert False
    except TypeError:
        failed_tests.append("Test 4 passed")

    #Test 5: incorrect diffusion coefficient
    try:
                            
            u, t = finite_difference(L, T, mx, mt, 'dirichlet', dirichlet_0, Initial_Condition, 
                                        discretisation='explicit',source_term = source_term, D = 'incorrect', linearity='linear')
                    
            assert False
    except TypeError:
        failed_tests.append("Test 5 passed")

    #Test 6: incorrect linearity
    try:
                                    
            u, t = finite_difference(L, T, mx, mt, 'dirichlet', dirichlet_0, Initial_Condition, 
                                            discretisation='explicit',source_term = source_term, D = D, linearity=[1,2])
                            
            assert False
    except UnboundLocalError:
        
        
        failed_tests.append("Test 6 passed")

    


    if len(failed_tests) == 0:
        print('\n---------------------------------------\n')
        print("All tests passed!")
        print('\n---------------------------------------\n')
    else:
        print('Some input tests failed:')
        for test in failed_tests:
            print('\n---------------------------------------\n')
            print(test)
            print('\n---------------------------------------\n')


import numpy as np

def test_PDE_output(tolerance):
    # Define the function to be solved
    L = 1  # length of the domain
    T = 0.5  # total time
    mx = 100  # number of grid points in space
    mt = 1000  # number of grid points in time

    def u_exact(x, t, D, L):
        y = np.exp(-D * (np.pi ** 2 / L ** 2) * t) * np.sin(np.pi * x / L)
        return y

    # Dirichlet boundary conditions, 0 on both ends.
    def dirichlet_0(x, t):
        return 0

    # Initial condition, sin(pi*x/L).
    def Initial_Condition(x, L):
        y = (np.sin(np.pi * x / L))
        return y

    # Diffusion coefficient, needs to be a function of x, to be compatible with the finite difference method.
    def D(x):
        return x / (x * 10)

    # Source term, needs to be a function of x and t, to be compatible with the finite difference method. 0 for this demonstration.
    def source_term(x, t):
        return 0

    # explicit euler and dirichlet
    u, t = finite_difference(L, T, mx, mt, 'dirichlet', dirichlet_0, Initial_Condition, discretisation='explicit',
                              source_term=source_term, D=D, linearity='linear')

    # Test the numerical solution against the true solution using allclose
    # set a tolerance value
    t_plot = t[-1] /2
    
    # Calculate the exact solution at the selected time step
    x = np.linspace(0, L, len(u))
    exact_solution = u_exact(x, t_plot,0.1,L)
    
    is_close = np.allclose(u[:, int(len(t) * t_plot / t[-1])], exact_solution, rtol=tolerance, atol=tolerance)
    print('\n---------------------------------------\n')
    print(f"Numerical and exact solutions match closely: {is_close}")
    print('\n---------------------------------------\n')




if __name__ == '__main__':
    test_ode_output(tolerance=1)
    test_shooting_output(tolerance=1e-3)
    test_shooting_inputs()
    test_shooting_generalised_inputs()
    test_ODE_inputs()
    test_PDE_inputs()
    test_PDE_output(tolerance=1e-3)
    #continuation_test()
# %%
