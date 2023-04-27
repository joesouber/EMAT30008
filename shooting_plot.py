from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import numdifftools as nd
from ODE import solve_ode


def shooting_generalised(f):
    """
    Returns a function G that defines the residual of a boundary value problem, suitable for use with a root-finding
    algorithm like scipy.optimize.root. The boundary value problem is defined by the function f, which represents a
    system of ordinary differential equations. The function G takes arguments u0_T, pc, *pars, where u0_T is a numpy
    array containing the initial state and the final time, pc is a function that computes the residuals at the boundary
    condition, and *pars are additional parameters that can be passed to the functions f and pc.
    
    Args:
        f: function defining the system of ODEs, takes arguments (t, x, *pars) where t is the current time, x is the
            current state, and *pars are additional parameters
            
    Returns:
        residual: function defining the residual of the boundary value problem, takes arguments u0_T, pc, *pars, where u0_T
            is a numpy array containing the initial state and the final time, pc is a function that computes the
            residuals at the boundary condition, and *pars are additional parameters that can be passed to the functions
            f and pc
    """

    def residual(u0_T, pc, *pars):
        """
        Computes the residual of a boundary value problem for a system of ODEs defined by the function f. The boundary
        value problem consists of finding the initial state u0 and final time T that satisfy the boundary conditions
        specified by the function pc. Additional parameters can be passed to the functions f and pc using *pars.
        
        Args:
            u0_T: numpy array containing the initial state and the final time, in that order
            pc: function that computes the residuals at the boundary condition, takes arguments (u0, *pars) where u0 is
                the initial state and *pars are additional parameters
            *pars: additional parameters to pass to the functions f and pc
            
        Returns:
            numpy array containing the residual of the boundary value problem, given u0_T and pc
        """
        
        # Create a list of time values to solve over
        t_eval = np.linspace(0, u0_T[-1], 1000)
        
        # Solve the ODE using solve_ode, and directly calculate the residuals
        return np.append(
            u0_T[:-1] - solve_ode(f, u0_T[:-1], t_eval, 0.01, 'RK4', True, *pars)[:, -1],
            pc(u0_T[:-1], *pars)
        )
    
    return residual

def plot_solutions(ode,pc,u0,args=(), max_step=0.01):
    
    def shooting(ode, u0, pc, solver, **params):
        assert callable(ode), "Error: 'ode' argument must be callable."
        #assert isinstance(u0, list) , "Error: 'u0' argument must be a list "
        assert callable(pc), "Error: 'pc' argument must be callable."
        assert callable(solver), "Error: 'solver' argument must be callable."
        G = shootingG(ode)
        orbit = solver(G, u0,pc, **params)
        print(orbit)
        return orbit

    def shootingG(ode):
        def G(x0, pc, **params):
            assert isinstance(x0, np.ndarray) and x0.ndim == 1, "Error: 'x0' argument must be a 1D numpy array."
            assert x0.size >= 2, "Error: 'x0' argument must have at least 2 elements."
            assert callable(pc), "Error: 'pc' argument must be callable."

            T = x0[-1]
            u0 = x0[:-1]
            assert isinstance(T, float) or isinstance(T, int), "Error: 'T' argument must be a float or an integer."

            # Directly calculate the residuals
            g = np.append(
                u0 - solve_ivp(ode, t_span=(0, T), y0=u0, method='RK45', **params).y[:, -1],
                pc(u0, **params)
            )

            assert isinstance(g, np.ndarray) and g.ndim == 1, "Error: 'g' must be a 1D numpy array."
            return g

        return G

    shooting_output = shooting(ode,u0, pc, fsolve)
    t_span = [0, shooting_output[-1]]
    # Extract the final time and initial guess from the shooting output
    T, u0 = shooting_output[-1], shooting_output[:-1]

    # Solve the ODE using solve_ivp with the initial guess from the shooting method
    sol = solve_ivp(ode, t_span=t_span, y0=u0, method='RK45', args=args, max_step=max_step)

    # Extract the predator and prey populations from the solution
    predator = sol.y[0]
    prey = sol.y[1]

    # Plot the predator and prey populations against time
    plt.plot(sol.t, predator, label='Predator')
    plt.plot(sol.t, prey, label='Prey')

    # Add labels and a legend
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()



def pred_prey_plot(a,d,bs,f):
    for b in bs:
        pars = [a, b, d]
        X0 = [0.5, 0.5]
        t_eval = np.linspace(0, 200, 3000)
        sol = solve_ivp(lambda t, X: f(X, pars), [0, 200], X0, t_eval=t_eval)


        plt.plot(sol.t, sol.y[0], label='Predator population')
        plt.plot(sol.t, sol.y[1], label='Prey population')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.show()



def plot_hopf_shooting(sol, pc, beta,normal_hopf):
    # Define the time vector.
    t = np.linspace(0, sol[-1], 2000)
    
    # Compute the phase angle.
    theta = pc(sol[:-1], beta)
    
    # Compute the analytical solutions.
    u1 = np.sqrt(beta) * np.cos(t + theta)
    u2 = np.sqrt(beta) * np.sin(t + theta)
    
    # Plot the analytical solutions.
    plt.plot(t, u1, label='exact solution 1', linestyle='--')
    plt.plot(t, u2, label='exact solution 2', linestyle='--')
    
    # Solve the system of equations using odeint.
    sol1 = odeint(normal_hopf, sol[:-1], t, args=(beta,))
    
    # Plot the numerical solutions.
    plt.plot(t, sol1[:, 0], label='numerical solution 1', linestyle=':')
    plt.plot(t, sol1[:, 1], label='numerical solution 2', linestyle=':')
    
    # Add labels and legend.
    plt.xlabel('t')
    plt.ylabel('u')
    plt.legend()
    
    # Show the plot.
    plt.show()