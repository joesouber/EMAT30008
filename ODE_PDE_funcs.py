import numpy as np

#first order ODE dx/dt = x
def f(x, t):
    return np.array([x])

#dx/dt = x, true solution
def f_true_solution(t):
    return np.exp(t)

