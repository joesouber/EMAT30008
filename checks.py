import numpy  as  np
def check_inputs_ODE(ODE, x0, t0, t1, method_name, h, system, *args):
    if not callable(ODE):
        raise TypeError("ODE must be a callable function")
    if not isinstance(x0, (int, float,list)):
        raise TypeError("x0 must be an integer, float or list if system")
    if not isinstance(t0, (int, float)):
        raise TypeError("t0 must be an integer or float")
    if not isinstance(t1, (int, float)):
        raise TypeError("t1 must be an integer or float")
    if not isinstance(method_name, str):
        raise TypeError("method_name must be a string")
    if method_name not in ['euler', 'RK4', 'heun']:
        raise ValueError("method_name must be either 'euler', 'RK4', or 'heun'")
    if not isinstance(h, (int, float)):
        raise TypeError("h must be an integer or float")
    if not isinstance(system, bool):
        raise TypeError("system must be a boolean")


def check_solution_accuracy(X, T, f_true_solution, rtol=1e-6, atol=1e-6):
    """
    Check the accuracy of the numerical solution by comparing it to the true solution.
    
    Parameters:
    X (ndarray): The numerical solution array of shape (n_steps+1, n), where n is the number of variables in the system.
    T (ndarray): The time array of shape (n_steps+1,) corresponding to the numerical solution.
    f_true_solution (callable): The function that computes the true solution.
    rtol (float): The relative tolerance parameter for np.allclose.
    atol (float): The absolute tolerance parameter for np.allclose.
    
    Returns:
    (bool): True if the numerical solution is close to the true solution, False otherwise.
    """
    X_true = np.array([f_true_solution(t) for t in T])
    return np.allclose(X, X_true, rtol=rtol, atol=atol)
