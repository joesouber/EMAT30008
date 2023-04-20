import numpy  as  np
#ODE
def check_callable(f):
    if not callable(f):
        raise TypeError("Input function is not callable")

def check_numeric(x0, max_step):
    if not isinstance(x0, (int, float)) or not isinstance(max_step, (int, float)):
        raise TypeError("Initial values and maximum step size must be numeric")

def check_pars(pars):
    if not isinstance(pars, tuple):
        raise TypeError("Input parameters must be a tuple")


# Output tests

X = solve_ode(f, x0, t_eval, max_step, method, system)
tolerance = 1e-9 # set a tolerance value

# Compare the first 100 elements of X and true_sol(t_eval)
is_close = np.allclose(X[:,0], true_sol(t_eval), rtol=tolerance, atol=tolerance)
print(f"Numerical and exact solutions match closely: {is_close}")



