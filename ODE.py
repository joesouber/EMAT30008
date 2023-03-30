#%%
import sys
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.integrate import solve_ivp

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


def solve_to(f, x1, t1, t2, method, deltat_max, *args):
    while t1 < t2:
        h = min(deltat_max, t2 - t1)
        x1, t1 = method(f, x1, t1, h, *args)
    return x1



def solve_ode(ODE, x0, t0, t1, method_name, deltat_max, second_order, *args):

    if method_name == 'Euler':
        method = euler_step
    elif method_name == 'RK4':
        method = RK4_step
    elif method_name == 'Heun':
        method = heun_step
    else:
        raise ValueError(
            f"The method '{method_name}' is not accepted, please try 'euler' or 'RK4'")

    number_steps = math.ceil(abs(t1 - t0) / deltat_max)

    
    if second_order:
        X = np.zeros((number_steps + 1, len(x0)))
    else:
        X = np.zeros((number_steps + 1, 1))

    T = np.zeros(number_steps + 1)
    X[0] = x0
    T[0] = t0

    for i in range(number_steps):

        if T[i] + deltat_max < t1:
            T[i + 1] = T[i] + deltat_max
        else:
            T[i + 1] = t1
        
        X[i + 1] = solve_to(ODE, X[i], T[i], T[i + 1], method, deltat_max, *args)

    return X, T



#%% need to test against the given ODEs, this should be in Jupyter notebook. 

def main():

    def f(x, t):
        return np.array([x])

    def FO_true_solution(t):
        return np.exp(t)

    
    method = 'Euler'
    f_euler, f_time = solve_ode(f, 1, 0, 1, method, 0.01, False)
    plt.plot(f_euler, f_time, label='Euler')

    method = 'RK4'
    f_RK4, f_time = solve_ode(f, 1, 0, 1, method, 0.01, False)
    plt.plot(f_RK4, f_time, label='RK4', linestyle='--')

    method = 'Heun'
    f_heun, f_time = solve_ode(f, 1, 0, 1, method, 0.01, False)
    plt.plot(f_heun, f_time, label='Heun')

    plt.plot(f_time, FO_true_solution(f_time), label='True', linestyle='--')

    plt.xlabel('t')
    plt.ylabel('x')
    plt.legend()
    plt.show()
# %%
main()
# %%
