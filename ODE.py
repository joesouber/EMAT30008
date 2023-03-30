#%%
import matplotlib.pyplot as plt
import math
import numpy as np
from ODE_PDE_funcs import f, f_true_solution

def euler_step(f, x0, t0, h, *args):

    x1 = x0 + h * f(x0, t0, *args)
    t1 = t0 + h

    return x1, t1

def RK4_step(f, x0, t0, h, *args):
   
    k1 = f(x0, t0, *args)
    k2 = f(x0 + h * 0.5 * k1, t0 + 0.5 * h, *args)
    k3 = f(x0 + h * 0.5 * k2, t0 + 0.5 * h, *args)
    k4 = f(x0 + h * k3, t0 + h, *args)

    k = 1 / 6 * h * (k1 + 2 * k2 + 2 * k3 + k4)

    x1 = x0 + k
    t1 = t0 + h

    return x1, t1


def heun_step(f, x0, t0, h, *args):
    k1 = f(x0, t0, *args)
    x1_tilde = x0 + h * k1
    k2 = f(x1_tilde, t0 + h, *args)

    k = 1 / 2 * h * (k1 + k2)
    x1 = x0 + k
    t1 = t0 + h

    return x1, t1


def solve_to(f, x1, t1, t2, method, h, *args):
    while t1 < t2:
        h = min(h, t2 - t1)
        x1, t1 = method(f, x1, t1, h, *args)
    return x1



def solve_ode(ODE, x0, t0, t1, method_name, h, system, *args):
    
    if method_name == 'euler':
        method = euler_step
    elif method_name == 'RK4':
        method = RK4_step
    elif method_name == 'heun':
        method = heun_step
    else:
        raise ValueError(
            f"The method '{method_name}' is not accepted, please try 'euler' or 'RK4'")
  
    number_steps = math.ceil(abs(t1 - t0) / h)
    if system:
        X = np.zeros((number_steps + 1, len(x0)))
    else:
        X = np.zeros((number_steps + 1, 1))
    T = np.zeros(number_steps + 1)
    X[0] = x0
    T[0] = t0
    for i in range(number_steps):
        if T[i] + h < t1:
            T[i + 1] = T[i] + h
        else:
            T[i + 1] = t1
        X[i + 1] = solve_to(ODE, X[i], T[i], T[i + 1], method, h, *args)
    return X, T

def calculate_error(ODE, true_solution, x0, t0, t1, method_name, deltat_list,system, *args):
    """
    Calculates the error between the approximate solution obtained using Euler, RK4 or Heun method and the true solution
    for different timesteps.

    Args:
        ODE (function): The ODE to be solved.
        true_solution (function): The true solution of the ODE.
        x0 (float): The initial value of x.
        t0 (float): The initial value of t.
        t1 (float): The final value of t.
        method_name (str): The name of the numerical method to be used. Must be 'Euler', 'RK4' or 'Heun'.
        deltat_list (list): A list of timesteps to use.
        args: Additional arguments for the ODE function.

    Returns:
        Tuple: Two lists containing the timesteps used and the corresponding errors.
    """
    if method_name == 'euler':
        method = euler_step
    elif method_name == 'RK4':
        method = RK4_step
    elif method_name == 'heun':
        method = heun_step
    else:
        raise ValueError(f"The method '{method_name}' is not accepted, please try 'euler','RK4'or 'heun'")

    errors = []
    timesteps = []
    for deltat in deltat_list:
        approx_solution, t = solve_ode(ODE, x0, t0, t1, method_name, deltat, system, *args)
        true_values = true_solution(t)
        errors.append(np.abs(approx_solution.flatten() - true_values.flatten()).max())
        timesteps.append(deltat)
    
    return timesteps, errors



def plot_error(ODE,ODE_true):

    deltat_list = [0.1, 0.05, 0.01, 0.005, 0.001]

    timesteps_euler, errors_euler = calculate_error(ODE, ODE_true, 1, 0, 1, 'euler', deltat_list,False)
    timesteps_rk4, errors_rk4 = calculate_error(ODE, ODE_true, 1, 0, 1, 'RK4', deltat_list,False)
    timesteps_heun, errors_heun = calculate_error(ODE, ODE_true, 1, 0, 1, 'heun', deltat_list,False)

    plt.loglog(timesteps_euler, errors_euler, 'o-', label='Euler')
    plt.loglog(timesteps_rk4, errors_rk4, 'o-', label='RK4')
    plt.loglog(timesteps_heun, errors_heun, 'o-', label='Heun')
    plt.xlabel('Timestep')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Errors for Numerical Methods')
    plt.show()



    





#%% This is to be written up in Jupyter.
def main():

    def f(x, t):
        return np.array([x])

    def f_true_solution(t):
        return np.exp(t)
    


    #method = 'euler'
    #f_euler, f_time = solve_ode(f, 1, 0, 1, method, 0.01, False)
    #plt.plot(f_time, f_euler, label='Euler')

    # Solve the ODE using the RK4 equation and plot the result
    #method = 'RK4'
    #f_RK4, f_time = solve_ode(f, 1, 0, 1, method, 0.01, False)
    #plt.plot(f_time, f_RK4, label='RK4')

    def f_ddot(u, t):
        x, y = u
        return np.array([y, -x])

    def f_ddot_true_solution(t):

        x = np.cos(t) + np.sin(t)
        y = np.cos(t) - np.sin(t)

        u = [x, y]

        return u

    # Solve the second order ODE using the RK4 equation
    method = 'RK4'
    ddot_rk4, ddot_time = solve_ode(f_ddot, [1, 1], 0, 10, method, 0.01, True)

    # RK4 and True solutions to the first initial condition
    plt.plot(ddot_time, ddot_rk4[:, 0], label='RK4')
    plt.plot(ddot_time, f_ddot_true_solution(ddot_time)[0], label='True',linestyle='--')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.legend()
    


# %%
#main()

plot_error(f, f_true_solution, x0, t0, t1, h_list)

