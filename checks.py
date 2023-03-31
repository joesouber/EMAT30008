
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
    