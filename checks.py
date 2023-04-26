import numpy  as  np


def check_callable(f):
    
    '''
    This function checks if the input function is callable.
    '''
    
    if not callable(f):
        raise TypeError("Input function is not callable")

def check_numeric(x0, max_step):
    
    '''
    This function checks if the initial values and maximum step size are numeric.
    '''

    if not isinstance(x0, (int, float)) or not isinstance(max_step, (int, float)):
        raise TypeError("Initial values and maximum step size must be numeric")

def check_pars(pars):
    
    '''
    This function checks if the input parameters are a tuple.
    '''

    if not isinstance(pars, tuple):
        raise TypeError("Input parameters must be a tuple")





