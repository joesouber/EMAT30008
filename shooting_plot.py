from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint



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
            def F(u0, T):
                tArr = np.linspace(0, T, 1000)
                sol = solve_ivp(ode, t_span=(0, T), y0=u0, method='RK45', **params)
                return sol.y[:, -1]
            T = x0[-1]
            u0 = x0[:-1]
            assert isinstance(T, float) or isinstance(T, int), "Error: 'T' argument must be a float or an integer."
            g = np.append(u0 - F(u0, T), pc(u0, **params))
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

## main

#%% Plots for Q1, might be useful for Jupyter.
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numdifftools as nd
from ODE import solve_ode
#%%



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

#pred_prey_plot(a,d,bs)

def shooting_generalised(f):
    def G(u0_T, pc, *pars):
        def F(u0, T):
            # Create a list of time values to solve over
            t_eval = np.linspace(0, T, 1000)
            sol =solve_ode(f, u0, t_eval, 0.01, 'RK4', True, *pars) 
            # Extract the final solution value
            final_sol = sol[:, -1]

            # return the final solution value
            return final_sol
        # Extract the inputted time and initial values
        T = u0_T[-1]
        u0= u0_T[:-1]
        return np.append(u0 - F(u0, T), pc(u0, *pars))
    return G

def plot_hopf_shooting(sol, pc, beta,normal_hopf):
    # Define the time vector.
    t = np.linspace(0, sol[-1], 2000)
    
    # Compute the phase angle.
    theta = pc(sol[:-1], beta)
    
    # Compute the analytical solutions.
    u1 = np.sqrt(beta) * np.cos(t + theta)
    u2 = np.sqrt(beta) * np.sin(t + theta)
    
    # Plot the analytical solutions.
    plt.plot(t, u1, label='u1', linestyle='--')
    plt.plot(t, u2, label='u2', linestyle='--')
    
    # Solve the system of equations using odeint.
    sol1 = odeint(normal_hopf, sol[:-1], t, args=(beta,))
    
    # Plot the numerical solutions.
    plt.plot(t, sol1[:, 0], label='numerical-solution 1', linestyle=':')
    plt.plot(t, sol1[:, 1], label='numsol2', linestyle=':')
    
    # Add labels and legend.
    plt.xlabel('t')
    plt.ylabel('u')
    plt.legend()
    
    # Show the plot.
    plt.show()