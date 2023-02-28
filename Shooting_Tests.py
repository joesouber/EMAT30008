import numpy as np
import week15
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# ode to solve using shooting
def hopf(t, u):

    u1, u2 = u
    beta = 1
    sigma = -1

    du1dt = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    du2dt = u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)

    return np.array((du1dt, du2dt))



# exact solution to differential equation
def true_hopf(t):
    beta = 1
    theta = 0.0

    u1 = np.sqrt(beta) * np.cos(t + theta)
    u2 = np.sqrt(beta) * np.sin(t + theta)

    return np.array([u1, u2])



# phase condition function for the hopf
def pc(u0, *args):
    return hopf(0, u0, *args)[0]


def test_limit_cycle(test_f, true_f, U0, pc, tolerance, endpoints,*args):
    """
    Function testing limit cycle solutions against the true value at a given tolerance, 
    can be specified to output the closeness of the initial conditions and endpoints of the 
    limit cycle to test if a full loop is formed"""

    # find limit cycle conditions of given function to test
    soll = week15.shooting(test_f, U0, pc,fsolve, *args)
     
    # unpack solution
    u0 = soll[:-1]
    T = soll[-1]

    # Use solve_ivp to get limit cycle values
    sol = solve_ivp(test_f,(0,T),u0)
    t_sol = sol.t
    u_sol = sol.y

    # transpose array to fit true_f dimensions
    u_sol = u_sol.T

    # get array shape to create according zero array
    rows, columns = u_sol.shape
    exact_sol = np.zeros([rows, columns])

    # get true solutions to the ode for the same times.
    for i in range(0,len(t_sol)):
        exact_sol[i] = true_f(t_sol[i])


    if endpoints == False:
        # compare true solutions to solutions obtained using the limit cycle.
        return np.allclose(u_sol, exact_sol, tolerance)
    elif endpoints == True:
        # compare initial conditions with limit cycle endpoints
        return np.allclose(exact_sol[0,:], exact_sol[-1,:], tolerance)
    


if __name__ == '__main__':

    print('\n')
    
    tolerance = np.logspace(-1,-21,21)  # tolerance range to test for
    
    # initial condition estimates
    U0_2d = (-1,0,6) 
    

    # test found limit cycle matches true values at same time points
    # test 2D ode
    for i in range(0,len(tolerance)):
        passed = test_limit_cycle(hopf,true_hopf,U0_2d,pc,tolerance[i],False)
        if passed == False:
            print(f"Limit cycle is found to be accurate at a tolerance level of {tolerance[i-1]} for a 2 dimensional ODE\n")
            break
        elif i == len(tolerance):
            print(f"Limit cycle is found to be accurate at a tolerance level of {tolerance[-1]} or greater, for a 2 dimensional ODE\n")

