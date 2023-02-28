

#%%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numdifftools as nd

def pred_prey_eq(X, pars):
  
    x = X[0]
    y = X[1]
    a, b, d = pars[0], pars[1], pars[2]
    dxdt = x * (1 - x) - (a * x * y) / (d + x)
    dydt = b * y * (1 - (y / x))
    return np.array([dxdt, dydt])

## simulating pred/prey equations uysing scipy solve_ivp. Should be able to switch this out for one of my functions eventually.

a = 1
d = 0.1
bs = [0.1,0.5]



for b in bs:
    pars = [a, b, d]
    X0 = [0.5, 0.5]
    t_eval = np.linspace(0, 200, 1000)
    sol = solve_ivp(lambda t, X: pred_prey_eq(X, pars), [0, 200], X0, t_eval=t_eval)


    plt.plot(sol.t, sol.y[0], label='Predator population')
    plt.plot(sol.t, sol.y[1], label='Prey population')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.show()

#To isolate a periodic orbit, we need to look for a closed trajectory in the phase space of the predator-prey system. One way to do this is to plot the phase portrait of the system, which shows the direction of motion of the system's trajectories. We can then identify any closed trajectories, which correspond to periodic orbits.

a = 1
d = 0.1
b= 0.5
pars = [a, b, d]

x = np.linspace(0, 1, 20)
y = np.linspace(0, 1, 20)
X, Y = np.meshgrid(x, y)
DX, DY = pred_prey_eq([X, Y], pars)
fig, ax = plt.subplots()
ax.streamplot(X, Y, DX, DY, color='red')
ax.set_xlabel('Prey population')
ax.set_ylabel('Predator population')

plt.show()


#We can see that there is a closed trajectory, or periodic orbit, in the phase space of the system. To find the starting conditions and period of this orbit, we can integrate the system starting from different initial conditions and look for orbits that close in on themselves.

X0 = [0.3, 0.3]
t_eval = np.linspace(0, 200, 1000)
sol = solve_ivp(lambda t, X: pred_prey_eq(X, pars), [0, 200], X0, t_eval=t_eval)

fig, ax = plt.subplots()
ax.plot(sol.y[0], sol.y[1], color='red')
ax.streamplot(X, Y, DX, DY, color='gray')
ax.set_xlabel('Prey population')
ax.set_ylabel('Predator population')

plt.show()


#%% This should work!
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def shooting(ode, u0, pc, solver, *args):
    """
    A function that uses numerical shooting to find limit cycles of
    a specified ODE.

    Parameters
    ----------
    ode : function
        The ODE to apply shooting to. The ode function should take
        a single parameter (the state vector) and return the
        right-hand side of the ODE as a numpy.array.
    u0 : numpy.array
        An initial guess at the initial values for the limit cycle.
    pc : function
        A function that computes the phase condition for the ODE.
        The phase condition should take the initial state vector
        as input and return a scalar value.
    solver : function
        A numerical solver for finding the root of the shooting
        function. The solver should take a function (the shooting
        function), an initial guess for the root, and additional
        arguments to the shooting function as input, and return
        the root as a numpy.array.
    args : tuple
        Additional arguments to pass to the ODE and phase condition
        functions.

    Returns
    -------
    numpy.array
        Returns a numpy.array containing the corrected initial values
        for the limit cycle. If the numerical root finder failed, the
        returned array is empty.
    """

    G = shootingG(ode)
    orbit = solver(G, u0, args=(pc, *args))
    return orbit


def shootingG(ode):
    """
    A function that returns the shooting function for the specified
    ODE.

    Parameters
    ----------
    ode : function
        The ODE to apply shooting to. The ode function should take
        a single parameter (the state vector) and return the
        right-hand side of the ODE as a numpy.array.

    Returns
    -------
    function
        Returns a function that takes the initial guess (as a numpy.array),
        the phase condition function, and additional arguments as input,
        and returns the shooting function as a numpy.array.
    """

    def G(x0, pc, *args):
        """
        The shooting function for the specified ODE.

        Parameters
        ----------
        x0 : numpy.array
            The initial guess for the limit cycle, with the last element
            being the period of the cycle.
        pc : function
            The phase condition function for the ODE.
        args : tuple
            Additional arguments to pass to the ODE and phase condition
            functions.

        Returns
        -------
        numpy.array
            Returns an array of size (n+1,), where n is the number of
            state variables in the ODE. The first n elements of the
            array are the difference between the initial guess and
            the ODE solution at the end of one period, and the last
            element is the difference between the phase condition
            and the value of the state variable that the phase
            condition should be equal to at the end of one period.
        """

        def F(u0, T):
            """
            A helper function that solves the ODE with the given
            initial values and time span.

            Parameters
            ----------
            u0 : numpy.array
                The initial values for the state variables of the ODE.
            T : float
                The time span to integrate over.

            Returns
            -------
            numpy.array
                Returns the final values of the state variables at
                the end of the time span.
            """
            tArr = np.linspace(0, T, 1000)
            sol = solve_ivp(ode, t_span=(0, T), y0=u0, method='RK45', args=tuple(args))
            return sol.y[:, -1]
        T = x0[-1]
        u0 = x0[:-1]
        g = np.append(u0 - F(u0, T), pc(u0, *args))  # Constructs array of ((initial guess - solution, phase condition)
        return g
    return G




def main():
    """
    Function for predator-prey equations
    """

    def predator_prey(t, y, args):
        x = y[0]
        y = y[1]

        a = args[0]
        d = args[1]
        b = args[2]
     
        dxdt = x * (1 - x) - (a * x * y) / (d + x)
        dydt = b * y * (1 - (y / x))
        return np.array([dxdt, dydt])
    
    def pc_predator_prey(u0, args):
        return predator_prey(0, u0, args)[0]
    
    predator_prey_u0 = np.array([0.07, 0.16, 23])
    
    args = [1, 0.1, 0.4]
    tArr = np.linspace(0, 1000, 1000)
    t = np.linspace(0, 1000, 1000)
    predator_prey_solution = solve_ivp(predator_prey, [tArr[0], tArr[-1]], predator_prey_u0[:-1], method='RK45', args=(args,))
    
    predator_prey_solution_x = predator_prey_solution.y[0]
    predator_prey_solution_y = predator_prey_solution.y[1]
    
    plt.plot(predator_prey_solution_x, predator_prey_solution_y,label='ODE')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.legend()
    plt.show()


    shooting_orbit = shooting(predator_prey, predator_prey_u0, pc_predator_prey, fsolve, args)
    
    plt.plot(shooting_orbit[0], shooting_orbit[1], 'ro', label="Numerical Shooting Orbit")
    plt.plot(predator_prey_solution_x, predator_prey_solution_y)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.legend()
    plt.show()

main()

