#%%
import matplotlib.pyplot as plt
import math
import numpy as np
from ODE_PDE_funcs import f, f_true_solution
from checks import check_inputs_ODE


def euler_step(f, x0, t0, max_step, *pars):


    x1 = x0 + max_step * f(x0, t0, *pars)
    t1 = t0 + max_step
    return x1, t1

def RK4_step(f, x0, t0, max_step, *pars):
   
    k1 = f(x0, t0, *pars)
    k2 = f(x0 + max_step * 0.5 * k1, t0 + 0.5 * max_step, *pars)
    k3 = f(x0 + max_step * 0.5 * k2, t0 + 0.5 * max_step, *pars)
    k4 = f(x0 + max_step * k3, t0 + max_step, *pars)

    k = 1 / 6 * max_step * (k1 + 2 * k2 + 2 * k3 + k4)

    x1 = x0 + k
    t1 = t0 + max_step

    return x1, t1



def heun_step(f, x0, t0, max_step, *pars):
    k1 = f(x0, t0, *pars)
    x1_tilde = x0 + max_step * k1
    k2 = f(x1_tilde, t0 + max_step, *pars)

    k = 1 / 2 * max_step * (k1 + k2)
    x1 = x0 + k
    t1 = t0 + max_step

    return x1, t1

def solve_to(f, x0, t0, t1, max_step, method='RK4', *pars):
    
    while t0 < t1:
        max_step = min(max_step, t1 - t0)  # limit step size to avoid overshooting t1
        if method in('euler','Euler'):
            x0, t0 = euler_step(f, x0, t0, max_step, *pars)
        elif method in ('RK4','rk4','Runge-Kutta'):
            x0, t0 = RK4_step(f, x0, t0, max_step, *pars)
        elif method in ('heun','Heun','h','Hun'):
            x0, t0 = heun_step(f,x0,t0,max_step,*pars)
    return x0




def solve_ode(f, x0, t_eval, max_step, method, system, *pars):

#insert name check function.


    # Define the empty x array depending on size of x0 and t_eval
    if system:
        X = np.zeros((len(t_eval), len(x0)))
    else:  # If it isn't a system of ODEs then len would return an error
        X = np.zeros((len(t_eval), 1))
    X[0] = x0
    
    steps = len(t_eval) - 1
    for i in range(steps):
        T_i = t_eval[i]
        T_ip1 = t_eval[i + 1]
        X_solved = solve_to(f, X[i], T_i, T_ip1,max_step, method, *pars)
        X[i+1]=X_solved
    if system: 
        X = X.transpose()
    
    return X
