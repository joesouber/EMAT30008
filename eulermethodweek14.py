#%% defining Euler method

def euler_step(x, t, dt, f):
    x_n_1 = x + dt * f(x, t)
    return x_n_1