import numpy  as  np
#ODE
def check_callable(f):
    if not callable(f):
        raise TypeError("Input function is not callable")

def check_numeric(x0, t0, max_step):
    if not isinstance(x0, (int, float)) or not isinstance(t0, (int, float)) or not isinstance(max_step, (int, float)):
        raise TypeError("Initial values and maximum step size must be numeric")

def check_pars(pars):
    if not isinstance(pars, tuple):
        raise TypeError("Input parameters must be a tuple")

def euler_step(f, x0, t0, max_step, *pars):
    check_callable(f)
    check_numeric(x0, t0, max_step)
    check_pars(pars)
    x1 = x0 + max_step * f(x0, t0, *pars)
    t1 = t0 + max_step
    return x1, t1



