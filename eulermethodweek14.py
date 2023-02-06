#%%
import numpy as np
import math

#%% defining Euler method

def euler_step(x, t, dt, f):
    x_n_1 = x + dt * f(x, t)
    return x_n_1
#%% making solve_to function to solve ODE between t0 and t1, with initial condition x0.

def solve_to(x0,t0,t1,deltat_max,f,method=euler_step):

#defining timesteps and then time loop with update rule.
    min_number_steps = math.floor((t1- t0) / deltat_max)
    
    #initialising arrays 
    x = np.array([])
    t = np.array([])
    for i in range(min_number_steps):
        t[i + 1] = t[i] + deltat_max
        x[i + 1] = euler_step(x[i], t[i], deltat_max,f)
    return x, t



# %%
