

#%% Plots for Q1, might be useful for Jupyter.
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numdifftools as nd
from ODE import solve_ode
#%%


#To isolate a periodic orbit, we need to look for a closed trajectory in the phase space of the predator-prey system. One way to do this is to plot the phase portrait of the system, which shows the direction of motion of the system's trajectories. We can then identify any closed trajectories, which correspond to periodic orbits.
#%%
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
    G = shootingG(ode)
    orbit = solver(G, u0, args=(pc, *args))
    print(orbit)
    return orbit

def shootingG(f):
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

#%%


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
    
    predator_prey_u0 = np.array([0.8, 0.2, 30])
    
    args = [1, 0.1, 0.1]
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
    plt.plot(predator_prey_solution_x, predator_prey_solution_y,linestyle='dashed', label='ODE')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.legend()
    plt.show()

main()

    
# %%
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
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
    
predator_prey_u0 = np.array([0.8, 0.2, 30])
    
args = [1, 0.1, 0.1]
tArr = np.linspace(0, 1000, 1000)
t = np.linspace(0, 1000, 1000)


shooting(predator_prey, predator_prey_u0, pc_predator_prey, fsolve, args)

# %%
